import numpy as np
from scipy import stats
from scipy.stats.tests.common_tests import (
    check_normalization, check_moment, check_mean_expect, check_var_expect,
    check_skew_expect, check_kurt_expect, check_edge_support,
    check_random_state_property, check_pickling)

from skypy.stats import schechter, genschechter


def test_schechter():
    # freeze a distribution with parameters
    args = (-1.2, 1e-5)
    dist = schechter(*args)

    # check that PDF is normalised
    check_normalization(schechter, args, 'schechter')

    # check CDF and SF
    assert np.isclose(dist.cdf(3.) + dist.sf(3.), 1.)

    # check inverse CDF and SF
    assert np.isclose(dist.ppf(dist.cdf(4.)), 4.)
    assert np.isclose(dist.isf(dist.sf(5.)), 5.)

    # check moments
    m, v, s, k = dist.stats(moments='mvsk')
    print(schechter._munp(2, *args))
    check_mean_expect(schechter, args, m, 'schechter')
    check_var_expect(schechter, args, m, v, 'schechter')
    check_skew_expect(schechter, args, m, v, s, 'schechter')
    check_kurt_expect(schechter, args, m, v, k, 'schechter')
    check_moment(schechter, args, m, v, 'schechter')

    # check other properties
    check_edge_support(schechter, args)
    check_random_state_property(schechter, args)
    check_pickling(schechter, args)

    # sample a single value
    rvs = dist.rvs()
    assert np.isscalar(rvs)

    # sample 10 values
    rvs = dist.rvs(size=10)
    assert rvs.shape == (10,)

    # sample with implicit sizes
    alpha, a = np.ones(5), np.ones((13, 7, 5))
    rvs = schechter.rvs(alpha, a)
    assert rvs.shape == np.broadcast(alpha, a).shape

    # check sampling against own CDF
    D, p = stats.kstest(schechter.rvs, schechter.cdf, args=args, N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # for alpha > 0, a = 0, the distribution is amma
    alpha = np.random.rand()
    dist = schechter(alpha, 1e-100)
    D, p = stats.kstest(dist.rvs(size=1000), 'gamma', args=(alpha+1,))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)


def test_genschechter():
    # freeze a distribution with parameters
    args = (-1.2, 1.5, 1e-5)
    dist = genschechter(*args)

    # check that PDF is normalised
    check_normalization(genschechter, args, 'genschechter')

    # check CDF and SF
    assert np.isclose(dist.cdf(3.) + dist.sf(3.), 1.)

    # check inverse CDF and SF
    assert np.isclose(dist.ppf(dist.cdf(4.)), 4.)
    assert np.isclose(dist.isf(dist.sf(5.)), 5.)

    # check moments
    m, v, s, k = dist.stats(moments='mvsk')
    print(genschechter._munp(2, *args))
    check_mean_expect(genschechter, args, m, 'genschechter')
    check_var_expect(genschechter, args, m, v, 'genschechter')
    check_skew_expect(genschechter, args, m, v, s, 'genschechter')
    check_kurt_expect(genschechter, args, m, v, k, 'genschechter')
    check_moment(genschechter, args, m, v, 'genschechter')

    # check other properties
    check_edge_support(genschechter, args)
    check_random_state_property(genschechter, args)
    check_pickling(genschechter, args)

    # sample a single value
    rvs = dist.rvs()
    assert np.isscalar(rvs)

    # sample 10 values
    rvs = dist.rvs(size=10)
    assert rvs.shape == (10,)

    # sample with implicit sizes
    alpha, gamma, a = np.ones(5), np.ones((7, 5)), np.ones((13, 7, 5))
    rvs = genschechter.rvs(alpha, gamma, a)
    assert rvs.shape == np.broadcast(alpha, gamma, a).shape

    # check sampling against own CDF
    D, p = stats.kstest(genschechter.rvs, genschechter.cdf, args=args, N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # for alpha > 0, gamma > 0, a = 0, the distribution is gengamma
    alpha, gamma = np.random.rand(), np.random.rand()
    dist = genschechter(alpha, gamma, 1e-100)
    a, c = (alpha+1)/gamma, gamma
    D, p = stats.kstest(dist.rvs(size=1000), 'gengamma', args=(a, c))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
