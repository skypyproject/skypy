r"""Models of galaxy luminosities.

"""

import numpy as np

from ..utils.random import schechter
from ..utils import uses_default_cosmology, dependent_argument


__all__ = [
    'absolute_to_apparent_magnitude',
    'apparent_to_absolute_magnitude',
    'distance_modulus',
    'luminosity_in_band',
    'luminosity_from_absolute_magnitude',
    'absolute_magnitude_from_luminosity',
    'schechter_lf_magnitude',
]


def absolute_to_apparent_magnitude(absolute_magnitude, distance_modulus):
    '''Convert absolute to apparent magnitude.

    Parameters
    ----------
    absolute_magnitude : array_like
        Absolute magnitude M.
    distance_modulus : array_like
        Distance modulus DM.

    Returns
    -------
    apparent_magnitude : array_like
        Apparent magnitude M + DM.

    '''

    return np.add(absolute_magnitude, distance_modulus)


def apparent_to_absolute_magnitude(apparent_magnitude, distance_modulus):
    '''Convert apparent to absolute magnitude.

    Parameters
    ----------
    apparent_magnitude : array_like
        Apparent magnitude m.
    distance_modulus : array_like
        Distance modulus DM.

    Returns
    -------
    absolute_magnitude : array_like
        Absolute magnitude m - DM.

    '''

    return np.subtract(apparent_magnitude, distance_modulus)


@uses_default_cosmology
def distance_modulus(redshift, cosmology):
    '''Compute the distance modulus.

    Parameters
    ----------
    redshift : array_like
        Redshift of objects.
    cosmology : Cosmology, optional
        The cosmology from which the luminosity distance is taken. If not
        given, the default cosmology is used.

    Returns
    -------
    distmod : array_like
        The distance modulus m - M for each input redshift.
    '''

    return cosmology.distmod(redshift).value


luminosity_in_band = {
    'Lsun_U': 6.33,
    'Lsun_B': 5.31,
    'Lsun_V': 4.80,
    'Lsun_R': 4.60,
    'Lsun_I': 4.51,
}
'''Bandpass magnitude of reference luminosities.

These values can be used for conversion in `absolute_magnitude_from_luminosity`
and `luminosity_from_absolute_magnitude`. The `Lsun_{UBVRI}` values contain the
absolute AB magnitude of the sun in Johnson/Cousins bands from [1]_.

References
----------
.. [1] Christopher N. A. Willmer 2018 ApJS 236 47

'''


def luminosity_from_absolute_magnitude(absolute_magnitude, zeropoint=None):
    """Converts absolute magnitudes to luminosities.

    Parameters
    ----------
    absolute_magnitude : array_like
        Input absolute magnitudes.
    zeropoint : float or str, optional
        Zeropoint for the conversion. If a string is given, uses the reference
        luminosities from `luminosity_in_band`.

    Returns
    -------
    luminosity : array_like
        Luminosity values.

    """

    if zeropoint is None:
        zeropoint = 0.
    elif isinstance(zeropoint, str):
        if zeropoint not in luminosity_in_band:
            raise KeyError('unknown zeropoint `{}`'.format(zeropoint))
        zeropoint = -luminosity_in_band[zeropoint]

    return 10.**(-0.4*np.add(absolute_magnitude, zeropoint))


def absolute_magnitude_from_luminosity(luminosity, zeropoint=None):
    """Converts luminosities to absolute magnitudes.

    Parameters
    ----------
    luminosity : array_like
        Input luminosity.
    zeropoint : float, optional
        Zeropoint for the conversion. If a string is given, uses the reference
        luminosities from `luminosity_in_band`.

    Returns
    -------
    absolute_magnitude : array_like
        Absolute magnitude values.

    """

    if zeropoint is None:
        zeropoint = 0.
    elif isinstance(zeropoint, str):
        if zeropoint not in luminosity_in_band:
            raise KeyError('unknown zeropoint `{}`'.format(zeropoint))
        zeropoint = -luminosity_in_band[zeropoint]

    return -2.5*np.log10(luminosity) - zeropoint


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
