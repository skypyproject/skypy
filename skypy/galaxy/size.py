r"""Galaxy size module.

This modules computes the angular size of galaxies from their physical size.
"""

import numpy as np
from astropy import units

from skypy.utils import uses_default_cosmology


__all__ = [
    'angular_size',
    'early_type_lognormal',
    'late_type_lognormal',
    'linear_lognormal',
]


@uses_default_cosmology
def angular_size(physical_size, redshift, cosmology):
    """Angular size of a galaxy.
    This function transforms physical radius into angular distance, described
    in [1].

    Parameters
    ----------
    physical_size : astropy.Quantity
        Physical radius of galaxies in units of length.
    redshift : float
        Redshifts at which to evaluate the angular diameter distance.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.

    Returns
    -------
    angular_size : astropy.Quantity
        Angular distances in units of [rad] for a given radius.

    Examples
    --------
    >>> from astropy import units
    >>> from skypy.galaxy.size import angular_size
    >>> from astropy.cosmology import Planck15
    >>> r = angular_size(10*units.kpc, 1, Planck15)

    References
    ----------
    .. [1] D. W. Hogg, (1999), astro-ph/9905116.
    """

    distance = cosmology.angular_diameter_distance(redshift)
    angular_size = np.arctan(physical_size / distance)

    return angular_size


def late_type_lognormal(magnitude, alpha, beta, gamma, M0, sigma1, sigma2,
                        size=None):
    """Lognormal distribution for late-type galaxies.
    This function provides a lognormal distribution for the physical size of
    late-type galaxies, described by equations 12, 15 and 16 in [1].

    Parameters
    ----------
    magnitude : float or array_like.
        Galaxy magnitude at which evaluate the lognormal distribution.
    alpha, beta, gamma, M0: float
        Model parameters describing the mean size of galaxies in [kpc].
        Equation 15 in [1].
    sigma1, sigma2: float
        Parameters describing the standard deviation of the lognormal
        distribution for the physical radius of galaxies. Equation 16 in [1].
    size : int or tuple of ints, optional.
        Output shape. If the given shape is, e.g., (m, n, k),
        then m * n * k samples are drawn. If size is None (default),
        a single value is returned if mean and sigma are both scalars.
        Otherwise, np.broadcast(mean, sigma).size samples are drawn.

    Returns
    -------
    physical_size : numpy.ndarray or astropy.Quantity
        Physical distance for a given galaxy with a given magnitude, in [kpc].
        If size is None and magnitude is a scalar, a single sample is returned.
        If size is ns, different from None, and magnitude is scalar,
        shape is (ns,). If magnitude has shape (nm,) and size=None,
        shape is (nm,).

    Examples
    --------
    >>> import numpy as np
    >>> from skypy.galaxy import size
    >>> magnitude = -16.0
    >>> alpha, beta, gamma, M0 = 0.21, 0.53, -1.31, -20.52
    >>> sigma1, sigma2 = 0.48, 0.25
    >>> s = size.late_type_lognormal(magnitude, alpha, beta, gamma, M0,\
                                     sigma1, sigma2)


    References
    ----------
    ..[1] S. Shen, H.J. Mo, S.D.M. White, M.R. Blanton, G. Kauffmann, W. Voges,
        J. Brinkmann, I. Csabai, Mon. Not. Roy. Astron. Soc. 343, 978 (2003).
    """

    if size is None and np.shape(magnitude):
        size = np.shape(magnitude)

    r_bar = np.power(10, -0.4 * alpha * magnitude + (beta - alpha) *
                     np.log10(1 + np.power(10, -0.4 * (magnitude - M0)))
                     + gamma) * units.kpc

    sigma_lnR = sigma2 + (sigma1 - sigma2) /\
                         (1.0 + np.power(10, -0.8 * (magnitude - M0)))

    return r_bar * np.random.lognormal(sigma=sigma_lnR, size=size)


def early_type_lognormal(magnitude, a, b, M0, sigma1, sigma2, size=None):
    """Lognormal distribution for early-type galaxies.
    This function provides a lognormal distribution for the physical size of
    early-type galaxies, described by equations 12, 14 and 16 in [1].

    Parameters
    ----------
    magnitude : float or array_like.
        Galaxy magnitude at which evaluate the lognormal distribution.
    a, b : float
        Linear model parameters describing the mean size of galaxies,
        equation 3.14 in [1].
    sigma: float
        Standard deviation of the lognormal distribution for the
        physical radius of galaxies.
    size : int or tuple of ints, optional.
        Output shape. If the given shape is, e.g., (m, n, k),
        then m * n * k samples are drawn. If size is None (default),
        a single value is returned if mean and sigma are both scalars.
        Otherwise, np.broadcast(mean, sigma).size samples are drawn.

    Returns
    -------
    physical_size : ndarray or astropy.Quantity
        Physical distance for a given galaxy with a given magnitude, in [kpc].
        If size is None and magnitude is a scalar, a single sample is returned.
        If size is ns, different from None, and magnitude is scalar,
        shape is (ns,). If magnitude has shape (nm,) and size=None,
        shape is (nm,).

    Examples
    --------
    >>> import numpy as np
    >>> from skypy.galaxy import size
    >>> magnitude = -20.0
    >>> a, b, M0 = 0.6, -4.63, -20.52
    >>> sigma1, sigma2 = 0.48, 0.25
    >>> s = size.early_type_lognormal(magnitude, a, b, M0, sigma1, sigma2)


    References
    ----------
    ..[1] S. Shen, H.J. Mo, S.D.M. White, M.R. Blanton, G. Kauffmann, W. Voges,
        J. Brinkmann, I. Csabai, Mon. Not. Roy. Astron. Soc. 343, 978 (2003).
    """

    return late_type_lognormal(magnitude, a, a, b, M0, sigma1, sigma2,
                               size=size)


def linear_lognormal(magnitude, a_mu, b_mu, sigma, size=None):
    """Lognormal distribution with linear mean.
    This function provides a lognormal distribution for the physical size of
    galaxies with a linear mean, described by equation 3.14 in [1]. See also
    equation 14 in [2].

    Parameters
    ----------
    magnitude : float or array_like.
        Galaxy absolute magnitude at which evaluate the lognormal distribution.
    a_mu, b_mu : float
        Linear model parameters describing the mean size of galaxies,
        equation 3.14 in [1].
    sigma: float
        Standard deviation of the lognormal distribution for the
        physical radius of galaxies.
    size : int or tuple of ints, optional.
        Output shape. If the given shape is, e.g., (m, n, k),
        then m * n * k samples are drawn. If size is None (default),
        a single value is returned if mean and sigma are both scalars.
        Otherwise, np.broadcast(mean, sigma).size samples are drawn.

    Returns
    -------
    physical_size : numpy.ndarray or astropy.Quantity
        Physical distance for a given galaxy with a given magnitude, in [kpc].
        If size is None and magnitude is a scalar, a single sample is returned.
        If size is ns, different from None, and magnitude is scalar,
        shape is (ns,). If magnitude has shape (nm,) and size=None,
        shape is (nm,).

    Examples
    --------
    >>> import numpy as np
    >>> from skypy.galaxy import size
    >>> magnitude = -20.0
    >>> a_mu, b_mu, sigma =-0.24, -4.63, 0.4
    >>> s = size.linear_lognormal(magnitude, a_mu, b_mu, sigma)

    References
    ----------
    ..[1] J. Herbel, T. Kacprzak, A. Amara, A. Refregier, C.Bruderer and
           A. Nicola, JCAP 1708, 035 (2017).
    ..[2] S. Shen, H.J. Mo, S.D.M. White, M.R. Blanton, G. Kauffmann, W.Voges,
           J. Brinkmann, I.Csabai, Mon. Not. Roy. Astron. Soc. 343, 978 (2003).
    """

    return late_type_lognormal(magnitude, -a_mu / 0.4, -a_mu / 0.4,
                               b_mu, -np.inf, sigma, sigma, size=size)
