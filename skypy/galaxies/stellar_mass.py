"""Models of galaxy stellar mass.
"""

import numpy as np

from ..utils.random import schechter


__all__ = [
    'schechter_smf_mass',
]


def schechter_smf_mass(redshift, alpha, m_star, m_min, m_max, size=None,
                       resolution=1000):
    r""" Stellar masses following the Schechter mass function [1]_.

    Parameters
    ----------
    redshift : array_like
        Galaxy redshifts for which to sample magnitudes.
    alpha : float
        The alpha parameter in the Schechter stellar mass function.
    m_star : (nm,) array-like
        Characteristic stellar mass m_*.
    size: int, optional
         Output shape of stellar mass samples. If size is None and m_star
         is a scalar, a single sample is returned. If size is None and
         m_star is an array, an array of samples is returned with the same
         shape as m_star.
    m_min, m_max : float
        Lower and upper bounds for the stellar mass.
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

    References
    ----------
    .. [1] Mo, H., Van den Bosch, F., & White, S. (2010). Galaxy Formation and
        Evolution. Cambridge: Cambridge University Press.
        doi:10.1017/CBO9780511807244

    """

    # only alpha scalars supported at the moment
    if np.ndim(alpha) > 0:
        raise NotImplementedError('only scalar alpha is supported')

    if size is None and np.shape(redshift):
        size = np.shape(redshift)

    # convert m_min, m_max to units of m_star
    x_min = m_min / m_star
    x_max = m_max / m_star

    # sample masses
    m = schechter(alpha, x_min, x_max, resolution, size=size, scale=m_star)

    return m
