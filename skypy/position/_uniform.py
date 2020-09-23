'''Implementations of uniform distributions.'''

import numpy as np

from astropy import units
from astropy.coordinates import SkyCoord, SkyOffsetFrame


TWO_PI = 2*np.pi
PI_HALF = np.pi/2


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

    # get cosine of the opening angle from the area
    cos_alpha = 1 - area.to_value(units.sr)/TWO_PI

    # randomly sample points within opening angle
    cos_theta = np.random.uniform(cos_alpha, 1, size=size)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = np.random.uniform(0, TWO_PI, size=size)

    # rotate to longitude, latitude
    sin_sin, sin_cos = sin_theta*np.sin(phi), sin_theta*np.cos(phi)
    lon = np.arctan2(sin_sin, cos_theta)
    lat = PI_HALF - np.arctan2(np.sqrt(1 - sin_cos**2), -sin_cos)

    # construct random sky coordinates around centre
    coords = SkyCoord(lon, lat, unit=units.rad, frame=SkyOffsetFrame(origin=centre))

    # return converted to input frame
    return coords.transform_to(centre, merge_attributes=False)
