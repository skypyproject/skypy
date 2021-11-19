"""Galaxy morphology module.

This module provides facilities to sample the sizes and ellipticities of
galaxies.
"""

__all__ = [
    'angular_size',
    'beta_ellipticity',
    'early_type_lognormal_size',
    'late_type_lognormal_size',
    'linear_lognormal_size',
    'ryden04_ellipticity',
]


import numpy as np
from scipy import stats
from astropy import units
from ..utils import random


def angular_size(physical_size, redshift, cosmology):
    """Angular size of a galaxy.
    This function transforms physical radius into angular distance, described
    in [1]_.

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

    References
    ----------
    .. [1] D. W. Hogg, (1999), astro-ph/9905116.
    """

    distance = cosmology.angular_diameter_distance(redshift)
    angular_size = np.arctan(physical_size / distance)

    return angular_size


def beta_ellipticity(e_ratio, e_sum, size=None):
    r'''Galaxy ellipticities sampled from a reparameterized beta distribution.

    The ellipticities follow a beta distribution parameterized by
    :math:`e_{\rm ratio}` and :math:`e_{\rm sum}` as presented in [1]_ Section
    III.A.

    Parameters
    ----------
    e_ratio : array_like
        Mean ellipticity of the distribution, must be between 0 and 1.
    e_sum : array_like
        Parameter controlling the width of the distribution, must be positive.

    Notes
    -----
    The probability distribution function :math:`p(e)` for ellipticity
    :math:`e` is given by a beta distribution:

    .. math::

        p(e) \sim \frac{\Gamma(a+b)}{\Gamma(a) \Gamma(b)} x^{a-1} (1-x)^{b-1}

    for :math:`0 <= e <= 1`, :math:`a = e_{\rm sum} e_{\rm ratio}`,
    :math:`b = e_{\rm sum} (1 - e_{\rm ratio})`, :math:`0 < e_{\rm ratio} < 1`
    and :math:`e_{\rm sum} > 0`, where :math:`\Gamma` is the gamma function.

    References
    ----------
    .. [1] Kacprzak T., Herbel J., Nicola A. et al., arXiv:1906.01018

    '''

    # convert to beta distribution parameters
    a = e_sum * e_ratio
    b = e_sum * (1.0 - e_ratio)

    # sample from the beta distribution
    return np.random.beta(a, b, size)


def late_type_lognormal_size(magnitude, alpha, beta, gamma, M0, sigma1, sigma2,
                             size=None):
    """Lognormal size distribution for late-type galaxies.

    This function provides a lognormal distribution for the physical size of
    late-type galaxies, described by equations 12, 15 and 16 in [1]_.

    Parameters
    ----------
    magnitude : float or array_like.
        Galaxy magnitude at which evaluate the lognormal distribution.
    alpha, beta, gamma, M0: float
        Model parameters describing the mean size of galaxies in [kpc].
        (Equation 15).
    sigma1, sigma2: float
        Parameters describing the standard deviation of the lognormal
        distribution for the physical radius of galaxies. (Equation 16).
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

    References
    ----------
    .. [1] S. Shen, H.J. Mo, S.D.M. White, M.R. Blanton, G. Kauffmann, W. Voges,
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


def early_type_lognormal_size(magnitude, a, b, M0, sigma1, sigma2, size=None):
    """Lognormal size distribution for early-type galaxies.

    This function provides a lognormal distribution for the physical size of
    early-type galaxies, described by equations 12, 14 and 16 in [1]_.

    Parameters
    ----------
    magnitude : float or array_like.
        Galaxy magnitude at which evaluate the lognormal distribution.
    a, b : float
        Linear model parameters describing the mean size of galaxies,
        (Equation 14).
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

    References
    ----------
    .. [1] S. Shen, H.J. Mo, S.D.M. White, M.R. Blanton, G. Kauffmann, W. Voges,
        J. Brinkmann, I. Csabai, Mon. Not. Roy. Astron. Soc. 343, 978 (2003).
    """

    return late_type_lognormal_size(magnitude, a, a, b, M0, sigma1, sigma2,
                                    size=size)


def linear_lognormal_size(magnitude, a_mu, b_mu, sigma, size=None):
    """Lognormal size distribution with linear mean.

    This function provides a lognormal distribution for the physical size of
    galaxies with a linear mean, described by equation 3.14 in [1]_. See also
    equation 14 in [2]_.

    Parameters
    ----------
    magnitude : float or array_like.
        Galaxy absolute magnitude at which evaluate the lognormal distribution.
    a_mu, b_mu : float
        Linear model parameters describing the mean size of galaxies,
        (Equation 3.14).
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

    References
    ----------
    .. [1] J. Herbel, T. Kacprzak, A. Amara, A. Refregier, C.Bruderer and
           A. Nicola, JCAP 1708, 035 (2017).
    .. [2] S. Shen, H.J. Mo, S.D.M. White, M.R. Blanton, G. Kauffmann, W.Voges,
           J. Brinkmann, I.Csabai, Mon. Not. Roy. Astron. Soc. 343, 978 (2003).
    """

    return late_type_lognormal_size(magnitude, -a_mu / 0.4, -a_mu / 0.4,
                                    b_mu, -np.inf, sigma, sigma, size=size)


def ryden04_ellipticity(mu_gamma, sigma_gamma, mu, sigma, size=None):
    r'''Ellipticity distribution of Ryden (2004).

    The ellipticity is sampled by randomly projecting a 3D ellipsoid with
    principal axes :math:`A > B > C` [1]_. The distribution of the axis ratio
    :math:`\gamma = C/A` is a truncated normal with mean :math:`\mu_\gamma` and
    standard deviation :math:`\sigma_\gamma`. The distribution of
    :math:`\epsilon = \log(1 - B/A)` is truncated normal with mean :math:`\mu`
    and standard deviation :math:`\sigma`.

    Parameters
    ----------
    mu_gamma : array_like
        Mean of the truncated Gaussian for :math:`\gamma`.
    sigma_gamma : array_like
        Standard deviation for :math:`\gamma`.
    mu : array_like
        Mean of the truncated Gaussian for :math:`\epsilon`.
    sigma : array_like
        Standard deviation for :math:`\epsilon`.
    size : int or tuple of ints or None
        Size of the sample. If `None` the size is inferred from the parameters.

    Returns
    -------
    ellipticity: (size,) array_like
        Ellipticities sampled from the Ryden 2004 model.

    References
    ----------
    .. [1] Ryden B. S., 2004, ApJ, 601, 214

    '''

    # get size if not given
    if size is None:
        size = np.broadcast(mu_gamma, sigma_gamma, mu, sigma).shape

    # truncation for gamma standard normal
    a_gam = np.divide(np.negative(mu_gamma), sigma_gamma)
    b_gam = np.divide(np.subtract(1, mu_gamma), sigma_gamma)

    # truncation for log(epsilon) standard normal
    a_eps = -np.inf
    b_eps = np.divide(np.negative(mu), sigma)

    # draw gamma and epsilon from truncated normal -- eq.s (10)-(11)
    gam = stats.truncnorm.rvs(a_gam, b_gam, mu_gamma, sigma_gamma, size=size)
    eps = np.exp(stats.truncnorm.rvs(a_eps, b_eps, mu, sigma, size=size))

    # scipy 1.5.x bug: make scalar if size is empty
    if size == () and not np.isscalar(gam):  # pragma: no cover
        gam, eps = gam.item(), eps.item()

    # random projection of random triaxial ellipsoid
    q = random.triaxial_axis_ratio(1-eps, gam)

    # return the ellipticity
    return (1-q)/(1+q)
