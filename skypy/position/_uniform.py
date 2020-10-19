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
    '''Uniform distribution of points over healpix pixel.

    Draws randomly distributed points from the healpix pixel `ipix` for a map
    with a given `nside` parameter.

    Parameters
    ----------
    nside : int
        Healpix map `nside` parameter.
    ipix : int
        Healpix map pixel index.
    size : int
        Number of points to draw.
    nest : bool, optional
        If True assume ``NESTED`` pixel ordering, otherwise ``RING`` pixel
        ordering. Default is ``RING`` pixel ordering.

    Returns
    -------
    coords : `~astropy.coordinates.SkyCoord`
        Randomly distributed points over the healpix pixel.

    Warnings
    --------
    This function requires the ``healpy`` package.

    Examples
    --------
    See :ref:`User Documentation <skypy.position.uniform_in_pixel>`.

    '''

    from healpy import pix2ang, max_pixrad, nside2pixarea, ang2pix

    # get the centre of the healpix pixel as a SkyCoord
    centre_lon, centre_lat = pix2ang(nside, ipix, nest=nest, lonlat=True)
    centre = SkyCoord(centre_lon, centre_lat, unit=units.deg)

    # get the maximum radius of a healpix pixel in radian
    r = max_pixrad(nside)

    # use that radius as the aperture of a spherical area in steradian
    area = TWO_PI*(1 - np.cos(r))*units.sr

    # oversampling factor = 1/(probability of the draw)
    over = area.value/nside2pixarea(nside)

    # the array of longitudes and latitudes of the sample
    lon, lat = np.empty(0), np.empty(0)

    # rejection sampling over irregularly shaped healpix pixels
    miss = size
    while miss > 0:
        # get the coordinates in a circular aperture around centre
        sample = uniform_around(centre, area, int(miss*over))

        # get longitude and latitude of the sample
        sample_lon, sample_lat = sample.ra.deg, sample.dec.deg

        # accept those positions that are inside the correct pixel
        accept = ipix == ang2pix(nside, sample_lon, sample_lat, nest=nest, lonlat=True)

        # store the new positions
        lon = np.append(lon, np.extract(accept, sample_lon))
        lat = np.append(lat, np.extract(accept, sample_lat))
        miss = size - len(lon)

    # construct the coordinates
    return SkyCoord(lon[:size], lat[:size], unit=units.deg)
