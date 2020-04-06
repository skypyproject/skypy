from astropy.units import allclose, isclose
import numpy as np
from scipy import stats
from scipy.stats.tests.common_tests import (
    check_normalization, check_moment, check_mean_expect, check_var_expect,
    check_skew_expect, check_kurt_expect, check_edge_support,
    check_random_state_property, check_pickling)


def test_kacprzak():

    from skypy.galaxy.morphology import kacprzak

    a, b = np.random.lognormal(size=2)
    args = (a / (a + b), a + b)
    beta_dist = stats.beta(a, b)
    kacprzak_dist = kacprzak(*args)
    kacprzak_uniform = kacprzak(0.5, 2.0)
    kacprzak_arcsine = kacprzak(0.5, 1.0)

    x = np.linspace(0, 1, 100)

    check_normalization(kacprzak, args, 'kacprzak')
    check_edge_support(kacprzak, args)
    check_random_state_property(kacprzak, args)
    check_pickling(kacprzak, args)

    m, v, s, k = kacprzak_dist.stats(moments='mvsk')
    check_mean_expect(kacprzak, args, m, 'kacprzak')
    check_var_expect(kacprzak, args, m, v, 'kacprzak')
    check_skew_expect(kacprzak, args, m, v, s, 'kacprzak')
    check_kurt_expect(kacprzak, args, m, v, k, 'kacprzak')
    check_moment(kacprzak, args, m, v, 'kacprzak')

    assert allclose(kacprzak_dist.pdf(x), beta_dist.pdf(x))
    assert allclose(kacprzak_dist.logpdf(x), beta_dist.logpdf(x))
    assert allclose(kacprzak_dist.cdf(x), beta_dist.cdf(x))
    assert allclose(kacprzak_dist.logcdf(x), beta_dist.logcdf(x))
    assert allclose(kacprzak_dist.ppf(x), beta_dist.ppf(x))
    assert allclose(kacprzak_dist.sf(x), beta_dist.sf(x))
    assert allclose(kacprzak_dist.logsf(x), beta_dist.logsf(x))
    assert allclose(kacprzak_dist.isf(x), beta_dist.isf(x))
    assert isclose(kacprzak_dist.entropy(), beta_dist.entropy())
    assert isclose(kacprzak_dist.median(), beta_dist.median())
    assert isclose(kacprzak_dist.std(), beta_dist.std())
    assert allclose(kacprzak_dist.interval(x), beta_dist.interval(x))

    assert np.isscalar(kacprzak_dist.rvs())
    assert kacprzak_dist.rvs(size=10).shape == (10,)

    e_ratio = 0.5 * np.ones((13, 1, 5))
    e_sum = 0.5 * np.ones((7, 5))
    rvs = kacprzak.rvs(e_ratio, e_sum)
    assert rvs.shape == np.broadcast(e_ratio, e_sum).shape

    D, p = stats.kstest(kacprzak_dist.rvs, beta_dist.cdf, N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    D, p = stats.kstest(kacprzak_uniform.rvs, 'uniform', N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    D, p = stats.kstest(kacprzak_arcsine.rvs, 'arcsine', N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
