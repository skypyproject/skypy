'''Implementations of uniform distributions.'''

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord

TWO_PI = 2*np.pi


@units.quantity_input(area=units.sr)
def uniform_around(centre, area, size):
    '''Uniform distribution of points around location.

    Draws randomly distributed points from a circular region of the given area
    around the centre point.

    Parameters
    ----------
    centre : `~astropy.coordinates.SkyCoord`
        Centre of the sampling region.
    area : `~astropy.units.Quantity`
        Area of the sampling region as a `~astropy.units.Quantity` in units of
        solid angle.
    size : int
        Number of points to draw.

    Returns
    -------
    coords : `~astropy.coordinates.SkyCoord`
        Randomly distributed points around the centre. The coordinates are
        returned in the same frame as the input.

    Examples
    --------
    See :ref:`User Documentation <skypy.position.uniform_around>`.

    '''

    # get cosine of maximum separation from area
    cos_theta_max = 1 - area.to_value(units.sr)/TWO_PI

    # randomly sample points within separation
    theta = np.arccos(np.random.uniform(cos_theta_max, 1, size=size))
    phi = np.random.uniform(0, TWO_PI, size=size)

    # construct random sky coordinates around centre
    return centre.directional_offset_by(phi, theta)


def uniform_in_pixel(nside, ipix, size, nest=False):
    from healpy import max_pixrad, pix2ang, ang2pix

    # get the centre of the healpix pixel as a SkyCoord
    centre_lon, centre_lat = pix2ang(nside, ipix, nest=False, lonlat=True)
    centre = SkyCoord(centre_lon, centre_lat, unit=units.deg)

    # get the maximum radius of a healpix pixel in radian
    r = max_pixrad(nside, degrees=False)

    # use that radius as the aperture of a spherical area in steradian
    area = TWO_PI*(1 - np.cos(r))*units.sr

    # the array of longitudes and latitudes of the sample
    lon, lat = np.empty(size), np.empty(size)

    # rejection sampling over irregularly shaped healpix pixels
    n_pos = 0
    while n_pos < size:
        # get the coordinates in a circular aperture around centre
        sample = uniform_around(centre, area, size-n_pos)

        # get longitude and latitude of the sample
        sample_lon, sample_lat = sample.ra.deg, sample.dec.deg

        # accept those positions that are inside the correct pixel
        accept = ipix == ang2pix(nside, sample_lon, sample_lat, nest=nest, lonlat=True)

        # count how many positions were accepted
        n_new = n_pos + accept.sum()

        # store the new positions
        lon[n_pos:n_new] = sample_lon[accept]
        lat[n_pos:n_new] = sample_lat[accept]
        n_pos = n_new

    # construct the coordinates
    return SkyCoord(lon, lat, unit=units.deg)
