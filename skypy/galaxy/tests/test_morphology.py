from astropy.units import allclose, isclose
import numpy as np
from scipy import stats
from scipy.stats.tests.common_tests import (
    check_normalization, check_moment, check_mean_expect, check_var_expect,
    check_skew_expect, check_kurt_expect, check_edge_support,
    check_random_state_property, check_pickling)


def test_ellipticity_beta():

    from skypy.galaxy.morphology import ellipticity_beta

    a, b = np.random.lognormal(size=2)
    args = (a + b, a / (a + b))
    beta = stats.beta(a, b)
    kacprzak = ellipticity_beta(*args)
    kacprzak_uniform = ellipticity_beta(2, 0.5)
    kacprzak_arcsine = ellipticity_beta(1, 0.5)

    x = np.linspace(0, 1, 100)
    intervals = np.linspace(0, 1, 100)

    check_normalization(ellipticity_beta, args, 'ellipticity_beta')
    check_edge_support(ellipticity_beta, args)
    check_random_state_property(ellipticity_beta, args)
    check_pickling(ellipticity_beta, args)

    m, v, s, k = kacprzak.stats(moments='mvsk')
    check_mean_expect(ellipticity_beta, args, m, 'ellipticity_beta')
    check_var_expect(ellipticity_beta, args, m, v, 'ellipticity_beta')
    check_skew_expect(ellipticity_beta, args, m, v, s, 'ellipticity_beta')
    check_kurt_expect(ellipticity_beta, args, m, v, k, 'ellipticity_beta')
    check_moment(ellipticity_beta, args, m, v, 'ellipticity_beta')

    assert allclose(kacprzak.pdf(x), beta.pdf(x))
    assert allclose(kacprzak.logpdf(x), beta.logpdf(x))
    assert allclose(kacprzak.cdf(x), beta.cdf(x))
    assert allclose(kacprzak.logcdf(x), beta.logcdf(x))
    assert allclose(kacprzak.ppf(x), beta.ppf(x))
    assert allclose(kacprzak.sf(x), beta.sf(x))
    assert allclose(kacprzak.logsf(x), beta.logsf(x))
    assert allclose(kacprzak.isf(x), beta.isf(x))
    assert isclose(kacprzak.entropy(), beta.entropy())
    assert isclose(kacprzak.median(), beta.median())
    assert isclose(kacprzak.std(), beta.std())
    assert allclose(kacprzak.interval(intervals), beta.interval(intervals))

    assert np.isscalar(kacprzak.rvs())
    assert kacprzak.rvs(size=10).shape == (10,)

    e_sum = 0.5 * np.ones((7, 5))
    e_ratio = 0.5 * np.ones((13, 1, 5))
    rvs = ellipticity_beta.rvs(e_sum, e_ratio)
    assert rvs.shape == np.broadcast(e_sum, e_ratio).shape

    D, p = stats.kstest(kacprzak.rvs, beta.cdf, N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    D, p = stats.kstest(kacprzak_uniform.rvs, 'uniform', N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    D, p = stats.kstest(kacprzak_arcsine.rvs, 'arcsine', N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
