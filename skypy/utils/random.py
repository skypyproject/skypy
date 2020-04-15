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
import skypy.utils.special as special


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

    x = np.logspace(np.log10(np.min(x_min)), np.log10(np.max(x_max)),
                    resolution)
    cdf = _schechter_cdf(x, np.min(x_min), np.max(x_max), alpha)
    t_lower = np.interp(x_min, x, cdf)
    t_upper = np.interp(x_max, x, cdf)
    u = np.random.uniform(t_lower, t_upper, size=size)
    x_sample = np.interp(u, cdf, x)

    return x_sample


def _schechter_cdf(x, x_min, x_max, alpha):
    a = special.upper_incomplete_gamma(alpha + 1, x_min)
    b = special.upper_incomplete_gamma(alpha + 1, x)
    c = special.upper_incomplete_gamma(alpha + 1, x_min)
    d = special.upper_incomplete_gamma(alpha + 1, x_max)

    return (a - b) / (c - d)
