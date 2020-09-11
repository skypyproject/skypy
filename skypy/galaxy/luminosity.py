r"""Models of galaxy luminosities.

"""

import numpy as np

from ..utils.random import schechter
from ..utils import uses_default_cosmology, dependent_argument


__all__ = [
    'absolute_to_apparent_magnitude',
    'apparent_to_absolute_magnitude',
    'distance_modulus',
    'schechter_lf_magnitude',
]


def absolute_to_apparent_magnitude(M, DM):
    '''Convert absolute to apparent magnitude.

    Parameters
    ==========
    M : array_like
        Absolute magnitude.
    DM : array_like
        Distance modulus.

    Returns
    =======
    m : array_like
        Apparent magnitude M + DM.

    '''

    return np.add(M, DM)


def apparent_to_absolute_magnitude(m, DM):
    '''Convert apparent to absolute magnitude.

    Parameters
    ==========
    m : array_like
        Apparent magnitude.
    DM : array_like
        Distance modulus.

    Returns
    =======
    M : array_like
        Absolute magnitude m - DM.

    '''

    return np.subtract(m, DM)


@uses_default_cosmology
def distance_modulus(redshift, cosmology):
    '''Compute the distance modulus.

    Parameters
    ==========
    redshift : array_like
        Redshift of objects.
    cosmology : Cosmology, optional
        The cosmology from which the luminosity distance is taken. If not
        given, the default cosmology is used.

    Returns
    =======
    distmod : array_like
        The distance modulus m - M for each input redshift.
    '''

    return cosmology.distmod(redshift).value


@uses_default_cosmology
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
    cosmology : Cosmology, optional
        Cosmology object for converting apparent and absolute magnitudes. If
        no cosmology is given, the default cosmology is used.
    size : int, optional
        Explicit size for the sampling. If not given, one magnitude is sampled
        for each redshift.

    Returns
    -------
    magnitude : array_like
        Absolute magnitude sampled from a Schechter luminosity function for
        each input galaxy redshift.

    Examples
    --------

    Sample a number of blue (alpha = -1.3, M_star = -20.5) galaxy magnitudes
    brighter than m = 22.0 around redshift 0.5.

    >>> import numpy as np
    >>> from skypy.galaxy.luminosity import schechter_lf_magnitude
    >>> z = np.random.uniform(4.9, 5.1, size=20)
    >>> M = schechter_lf_magnitude(z, -20.5, -1.3, 22.0)

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
