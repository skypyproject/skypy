"""Galaxy ellipticity module.

This module provides facilities to sample the ellipticities of galaxies.
"""

import numpy as np
from scipy import stats


__all__ = [
    'beta_ellipticity',
    'ryden04',
]


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

    Examples
    --------
    Sample 10000 random variates from the Kacprzak model with
    :math:`e_{\rm ratio} = 0.5` and :math:`e_{\rm sum} = 1.0`:

    >>> from skypy.galaxy.ellipticity import beta_ellipticity
    >>> ellipticity = beta_ellipticity(0.5, 1.0, size=10000)

    '''

    # convert to beta distribution parameters
    a = e_sum * e_ratio
    b = e_sum * (1.0 - e_ratio)

    # sample from the beta distribution
    return np.random.beta(a, b, size)


def ryden04(mu_gamma, sigma_gamma, mu, sigma, size=None):
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

    References
    ----------
    .. [1] Ryden B. S., 2004, ApJ, 601, 214

    Examples
    --------
    Sample 10000 random variates from the Ryden (2004) model with parameters
    :math:`\mu_\gamma = 0.222`, :math:`\sigma_\gamma = 0.056`,
    :math:`\mu = -1.85`, and :math:`\sigma = 0.89`.

    >>> from skypy.galaxy.ellipticity import ryden04
    >>> ellipticity = ryden04(0.222, 0.056, -1.85, 0.89, size=10000)

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

    # draw random viewing angle (theta, phi)
    cos2_theta = np.random.uniform(low=-1., high=1., size=size)**2
    cos2_phi = np.cos(np.random.uniform(low=0., high=2*np.pi, size=size))**2
    sin2_theta = 1 - cos2_theta
    sin2_phi = 1 - cos2_phi

    # compute Binney's ABC -- eq.s (13)-(15)
    A = (1 - eps*(2-eps)*sin2_phi)*cos2_theta + gam**2*sin2_theta
    B = (2*eps*(2-eps))**2*cos2_theta*sin2_phi*cos2_phi
    C = 1 - eps*(2-eps)*cos2_phi

    # compute axis ratio q -- eq. (12)
    q = np.sqrt((A+C-np.sqrt((A-C)**2+B))/(A+C+np.sqrt((A-C)**2+B)))

    # return the ellipticity
    return (1-q)/(1+q)
