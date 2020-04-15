r'''Halo mass module.

This module provides methods to sample the  masses of dark matter halos.

Models
======

.. autosummary::

   :nosignatures:
   :toctree: ../api/

   press_schechter

'''

import numpy as np

from skypy.utils.random import schechter


def press_schechter(n, m_star, size=None, x_min=0.00305,
                    x_max=1100.0, resolution=100):
    """Sampling from Press-Schechter mass function (1974).

    Masses following the Press-Schechter mass function following the
    Press et al. [1]_ formalism.

    Parameters
    ----------
    n : float or int
        The n parameter in the Press-Schechter mass function.
    m_star : float or int
        Factors parameterising the characteristic mass.
    size: int, optional
         Output shape of luminosity samples.
    x_min, x_max : float or int, optional
        Lower and upper bounds in units of M*.
    resolution : int, optional
        Resolution of the inverse transform sampling spline. Default is 100.

    Returns
    -------
    mass : array_like
        Drawn masses from the Press-Schechter mass function.

    Examples
    --------
    >>> import skypy.halo.mass as mass
    >>> n, m_star = 1, 1e9
    >>> sample = mass.press_schechter(n, m_star, size=1000, x_min=1e-10,
    ...                               x_max=1e2, resolution=1000)

    References
    ----------
    .. [1] Press, W. H. and Schechter, P., APJ, (1974).

    """

    alpha = - 0.5 * (n + 9.0) / (n + 3.0)

    x_sample = schechter(alpha, x_min, x_max, resolution=resolution, size=size)

    return m_star * np.power(x_sample, 3.0 / (n + 3.0))
