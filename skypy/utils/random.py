import numpy as np
import skypy.utils.special as special


def schechter(alpha, x_min, x_max, resolution=100, size=None):
    """Galaxy sampling from the Schechter function.

    Parameters
    ----------
    alpha : float or int
        The alpha parameter in the Schechter function in [1]_.
    x_min, x_max : float or int
        Lower and upper bounds for the random variable x.
    resolution : int
        Resolution of the inverse transform sampling spline. Default is 100.
    size: int, optional
         Output shape of luminosity samples.  Default is None.

    Returns
    -------
    x_sample : array_like
        Drawn sampling from the Schechter function.

    References
    ----------
    .. [1] Herbel J., Kacprzak T., Amara A. et al., 2017, Journal of Cosmology
           and Astroparticle Physics, Issue 08, article id. 035 (2017)

    """
    x = np.logspace(np.log10(np.min(x_min)), np.log10(np.max(x_max)),
                    resolution)
    cdf = _cdf(x, np.min(x_min), np.max(x_max), alpha)
    t_lower = np.interp(x_min, x, cdf)
    t_upper = np.interp(x_max, x, cdf)
    u = np.random.uniform(t_lower, t_upper, size=size)
    x_sample = np.interp(u, cdf, x)

    return x_sample


def _cdf(x, x_min, x_max, alpha):
    a = special.upper_incomplete_gamma(alpha + 1, x_min)
    b = special.upper_incomplete_gamma(alpha + 1, x)
    c = special.upper_incomplete_gamma(alpha + 1, x_min)
    d = special.upper_incomplete_gamma(alpha + 1, x_max)

    return (a - b) / (c - d)
