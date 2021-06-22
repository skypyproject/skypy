r"""Models of galaxy luminosities.

"""

import numpy as np

from ..utils.random import schechter
from ..utils import dependent_argument


__all__ = [
    'schechter_lf_magnitude',
]


@dependent_argument('M_star', 'redshift')
@dependent_argument('alpha', 'redshift')
def schechter_lf_magnitude(redshift, M_star, alpha, m_lim, cosmology, size=None,
                           x_max=1e3, resolution=1000):
    r'''Sample magnitudes from a Schechter luminosity function.

    Given a list of galaxy redshifts, and an apparent magnitude limit, sample
    galaxy absolute magnitudes from a Schechter luminosity function.

    Parameters
    ----------
    redshift : array_like
        Galaxy redshifts for which to sample magnitudes.
    M_star : array_like or function
        Characteristic absolute magnitude, either constant, or an array with
        values for each galaxy, or a function of galaxy redshift.
    alpha : array_like or function
        Schechter function index, either a constant, or an array of values for
        each galaxy, or a function of galaxy redshift.
    m_lim : float
        Apparent magnitude limit.
    cosmology : Cosmology
        Cosmology object for converting apparent and absolute magnitudes.
    size : int, optional
        Explicit size for the sampling. If not given, one magnitude is sampled
        for each redshift.

    Returns
    -------
    magnitude : array_like
        Absolute magnitude sampled from a Schechter luminosity function for
        each input galaxy redshift.

    '''

    # only alpha scalars supported at the moment
    if np.ndim(alpha) > 0:
        raise NotImplementedError('only scalar alpha is supported')

    # get x_min for each galaxy
    x_min = m_lim - cosmology.distmod(redshift).value
    x_min -= M_star
    x_min *= -0.4
    if np.ndim(x_min) > 0:
        np.power(10., x_min, out=x_min)
    else:
        x_min = 10.**x_min

    # sample magnitudes
    #   x == 10**(-0.4*(M - M_star))
    M = schechter(alpha, x_min, x_max, resolution, size=size)
    np.log10(M, out=M)
    M *= -2.5
    M += M_star

    return M
