from astropy.units import allclose, isclose
import numpy as np
from scipy import stats
from scipy.stats.tests.common_tests import (
    check_normalization, check_moment, check_mean_expect, check_var_expect,
    check_skew_expect, check_kurt_expect, check_edge_support,
    check_random_state_property, check_pickling)


def test_beta_ellipticity():

    from skypy.galaxy.ellipticity import beta_ellipticity

    # Initialise a randomised ellipticity distribution, an equivalent beta
    # distribution and special cases where the ellipticity distribution is
    # equivalent to a uniform distribution and an arcsine distribution.
    a, b = np.random.lognormal(size=2)
    args = (a / (a + b), a + b)
    beta_dist = stats.beta(a, b)
    ellipticity_dist = beta_ellipticity(*args)
    ellipticity_uniform = beta_ellipticity(0.5, 2.0)
    ellipticity_arcsine = beta_ellipticity(0.5, 1.0)

    # Range of input values spanning the support of the distributions
    x = np.linspace(0, 1, 100)

    # Check basic properties of distribution implementation
    check_normalization(beta_ellipticity, args, 'beta_ellipticity')
    check_edge_support(beta_ellipticity, args)
    check_random_state_property(beta_ellipticity, args)
    check_pickling(beta_ellipticity, args)

    # Check distribution moments
    m, v, s, k = ellipticity_dist.stats(moments='mvsk')
    check_mean_expect(beta_ellipticity, args, m, 'beta_ellipticity')
    check_var_expect(beta_ellipticity, args, m, v, 'beta_ellipticity')
    check_skew_expect(beta_ellipticity, args, m, v, s, 'beta_ellipticity')
    check_kurt_expect(beta_ellipticity, args, m, v, k, 'beta_ellipticity')
    check_moment(beta_ellipticity, args, m, v, 'beta_ellipticity')

    # Compare ellipticity distribution functions (e.g. pdf, cdf...) against
    # functions for an equivalent beta distribution
    assert allclose(ellipticity_dist.pdf(x), beta_dist.pdf(x))
    assert allclose(ellipticity_dist.logpdf(x), beta_dist.logpdf(x))
    assert allclose(ellipticity_dist.cdf(x), beta_dist.cdf(x))
    assert allclose(ellipticity_dist.logcdf(x), beta_dist.logcdf(x))
    assert allclose(ellipticity_dist.ppf(x), beta_dist.ppf(x))
    assert allclose(ellipticity_dist.sf(x), beta_dist.sf(x))
    assert allclose(ellipticity_dist.logsf(x), beta_dist.logsf(x))
    assert allclose(ellipticity_dist.isf(x), beta_dist.isf(x))
    assert isclose(ellipticity_dist.entropy(), beta_dist.entropy())
    assert isclose(ellipticity_dist.median(), beta_dist.median())
    assert isclose(ellipticity_dist.std(), beta_dist.std())
    assert allclose(ellipticity_dist.interval(x), beta_dist.interval(x))

    # Test scalar output
    assert np.isscalar(ellipticity_dist.rvs())

    # Test array output
    assert ellipticity_dist.rvs(size=10).shape == (10,)

    # Test broadcast output
    e_ratio = 0.5 * np.ones((13, 1, 5))
    e_sum = 0.5 * np.ones((7, 5))
    rvs = beta_ellipticity.rvs(e_ratio, e_sum)
    assert rvs.shape == np.broadcast(e_ratio, e_sum).shape

    # Kolmogorov-Smirnov test comparing ellipticity and beta distributions
    D, p = stats.kstest(ellipticity_dist.rvs, beta_dist.cdf, N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # Kolmogorov-Smirnov test comparing ellipticity and uniform distributions
    D, p = stats.kstest(ellipticity_uniform.rvs, 'uniform', N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # Kolmogorov-Smirnov test comparing ellipticity and arcsine distributions
    D, p = stats.kstest(ellipticity_arcsine.rvs, 'arcsine', N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
