'''Implementations of uniform distributions.'''

import numpy as np
from astropy import units

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

    # get cosine of maximum separation from area
    cos_theta_max = 1 - area.to_value(units.sr)/TWO_PI

    # randomly sample points within separation
    theta = np.arccos(np.random.uniform(cos_theta_max, 1, size=size))
    phi = np.random.uniform(0, TWO_PI, size=size)

    # construct random sky coordinates around centre
    return centre.directional_offset_by(phi, theta)
