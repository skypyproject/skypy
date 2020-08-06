"""Random utility module.

This module provides methods to draw from random distributions.


Utility Functions
=================

.. autosummary::
   :nosignatures:
   :toctree: ../api/

   schechter

"""

import numpy as np


def schechter(alpha, x_min, x_max, resolution=100, size=None):
    """Sample from the Schechter function.

    Parameters
    ----------
    alpha : float or int
        The alpha parameter in the Schechter function in [1]_.
    x_min, x_max : array_like
        Lower and upper bounds for the random variable x.
    resolution : int
        Resolution of the inverse transform sampling spline. Default is 100.
    size: int, optional
        Output shape of samples. Default is None.

    Returns
    -------
    x_sample : array_like
        Samples drawn from the Schechter function.

    Examples
    --------
    >>> import skypy.utils.random as random
    >>> alpha = -1.3
    >>> sample = random.schechter(alpha, x_min=1e-10, x_max=1e2,
    ...                           resolution=100, size=1000)


    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Luminosity_function_(astronomy)
    """

    lnx_min = np.log(x_min)
    lnx_max = np.log(x_max)

    lnx = np.linspace(np.min(lnx_min), np.max(lnx_max), resolution)

    cdf = np.exp(np.add(alpha, 1)*lnx - np.exp(lnx))
    np.cumsum((cdf[1:]+cdf[:-1])/2*np.diff(lnx), out=cdf[1:])
    cdf[0] = 0
    cdf /= cdf[-1]

    t_lower = np.interp(lnx_min, lnx, cdf)
    t_upper = np.interp(lnx_max, lnx, cdf)
    u = np.random.uniform(t_lower, t_upper, size=size)
    lnx_sample = np.interp(u, cdf, lnx)

    return np.exp(lnx_sample)
