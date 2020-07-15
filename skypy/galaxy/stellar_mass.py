"""Models of galaxy stellar mass.
"""

import numpy as np

from skypy.utils.random import schechter


__all__ = [
    'schechter_smf',
]


def schechter_smf(alpha, m_star, x_min, x_max, resolution=100, size=None):
    r""" Stellar masses following the Schechter mass function [1]_.

    Parameters
    ----------
    alpha : float
        The alpha parameter in the Schechter stellar mass function.
    m_star : (nm,) array-like
        Characteristic stellar mass M_*.
    size: int, optional
         Output shape of stellar mass samples. If size is None and m_star
         is a scalar, a single sample is returned. If size is None and
         m_star is an array, an array of samples is returned with the same
         shape as m_star.
    x_min, x_max : float
        Lower and upper bounds for the random variable x in units of M_*.
    resolution : int, optional
        Resolution of the inverse transform sampling spline. Default is 100.

    Returns
    -------
    stellar mass : (nm,) array_like
        Drawn stellar masses from the Schechter stellar mass function in units
        of the solar mass.

    Notes
    -----
    The stellar mass probability distribution (pdf) follows a Schechter
    profile of the form

    .. math::

        \Phi(M) = \frac{1}{M_*} \left(\frac{M}{M_*}\right)^\alpha
            \exp\left(-\frac{M}{M_*}\right) \;.

    From this pdf one can sample the stellar masses.

    Examples
    --------
    >>> from skypy.galaxy import stellar_mass

    Sample 100 stellar masses values at redshift z = 1.0 with alpha = -1.4,
    m_star = 10**10.67, x_min = 0.0002138 and x_max = 213.8

    >>> masses = stellar_mass.schechter_smf(-1.4, 10**10.67, 0.0002138,
    ...                               213.8, size=100)

    References
    ----------
    .. [1] Mo, H., Van den Bosch, F., & White, S. (2010). Galaxy Formation and
        Evolution. Cambridge: Cambridge University Press.
        doi:10.1017/CBO9780511807244

    """

    if size is None and np.shape(m_star):
        size = np.shape(m_star)

    x_sample = schechter(alpha, x_min, x_max, resolution=resolution, size=size)

    return m_star * x_sample
