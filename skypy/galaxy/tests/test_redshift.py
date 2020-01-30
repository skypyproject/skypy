import numpy as np
from scipy import stats
from scipy.stats.tests.common_tests import (
    check_normalization, check_moment, check_mean_expect, check_var_expect,
    check_skew_expect, check_kurt_expect, check_edge_support,
    check_random_state_property, check_pickling)
import pytest

def test_smail():
    from skypy.galaxy.redshift import smail

    # freeze a distribution with parameters
    args = (1.3, 2.0, 1.5)
    dist = smail(*args)

    # check that PDF is normalised
    check_normalization(smail, args, 'smail')

    # check CDF and SF
    assert np.isclose(dist.cdf(3.) + dist.sf(3.), 1.)

    # check inverse CDF and SF
    assert np.isclose(dist.ppf(dist.cdf(4.)), 4.)
    assert np.isclose(dist.isf(dist.sf(5.)), 5.)

    # check moments
    m, v, s, k = dist.stats(moments='mvsk')
    check_mean_expect(smail, args, m, 'smail')
    check_var_expect(smail, args, m, v, 'smail')
    check_skew_expect(smail, args, m, v, s, 'smail')
    check_kurt_expect(smail, args, m, v, k, 'smail')
    check_moment(smail, args, m, v, 'smail')

    # check other properties
    check_edge_support(smail, args)
    check_random_state_property(smail, args)
    check_pickling(smail, args)

    # sample a single redshift
    rvs = dist.rvs()
    assert np.isscalar(rvs)

    # sample 10 reshifts
    rvs = dist.rvs(size=10)
    assert rvs.shape == (10,)

    # sample with implicit sizes
    zm, a, b = np.ones(5), np.ones((7, 5)), np.ones((13, 7, 5))
    rvs = smail.rvs(z_median=zm, alpha=a, beta=b)
    assert rvs.shape == np.broadcast(zm, a, b).shape

    # check sampling
    D, p = stats.kstest(smail.rvs, smail.cdf, args=args, N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # check sampling, for alpha=0, beta=1, the distribution is exponential
    D, p = stats.kstest(smail.rvs(0.69315, 1e-100, 1., size=1000) , 'expon')
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # check sampling, for beta=1, the distribution matches a gamma distribution
    D, p = stats.kstest(smail.rvs(2.674, 2, 1, size=1000), 'gamma', args=(3,))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
