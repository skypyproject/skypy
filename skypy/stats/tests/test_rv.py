import numpy as np
import scipy.stats
from astropy import log

from skypy.stats import *


_WRAPPED_METHODS = [
    'median',
    'mean',
    'var',
    'std',
    'entropy',
    'support',
]


_WRAPPED_METHODS_OF_X = [
    'pdf',
    'logpdf',
    'pmf',
    'logpmf',
    'cdf',
    'logcdf',
    'sf',
    'logsf',
]

_WRAPPED_METHODS_OF_Q = [
    'ppf',
    'isf',
]


def _test_wrap(dist, argsfn, wrap_args):
    log.debug('testing wrapped %s' % str(dist))

    # wrap the distribution
    wrap = parametrise(dist, argsfn)

    # arguments for distribution
    dist_args = argsfn(*wrap_args)

    # sample random number
    log.debug('... method `rvs`')
    np.random.seed(1234)
    x = dist.rvs(*dist_args)
    np.random.seed(1234)
    y = wrap.rvs(*wrap_args)
    assert x == y, 'wrapped distribution sampled different value'

    # test plain methods
    for m in _WRAPPED_METHODS:
        if hasattr(dist, m):
            log.debug('... method `%s`' % m)
            f = getattr(dist, m)
            g = getattr(wrap, m)
            a = f(*dist_args)
            b = g(*wrap_args)
            assert a == b, 'wrapped method `%s` produced different result' % m

    # test methods of x or k
    for m in _WRAPPED_METHODS_OF_X:
        if hasattr(dist, m):
            log.debug('... method `%s`' % m)
            f = getattr(dist, m)
            g = getattr(wrap, m)
            a = f(x, *dist_args)
            b = g(x, *wrap_args)
            assert a == b, 'wrapped method `%s` produced different result' % m

    # test methods of quantile
    q = np.random.rand()
    for m in _WRAPPED_METHODS_OF_Q:
        if hasattr(dist, m):
            log.debug('... method `%s`' % m)
            f = getattr(dist, m)
            g = getattr(wrap, m)
            a = f(q, *dist_args)
            b = g(q, *wrap_args)
            assert a == b, 'wrapped method `%s` produced different result' % m

    # compare stats
    log.debug('... method `stats`')
    dist_mvsk = dist.stats(*dist_args, moments='mvsk')
    wrap_mvsk = wrap.stats(*wrap_args, moments='mvsk')
    assert dist_mvsk == wrap_mvsk, 'wrapped stats are different'

    # compare a number of moments
    log.debug('... method `moment`')
    dist_moms = [dist.moment(n, *dist_args) for n in range(5)]
    wrap_moms = [wrap.moment(n, *wrap_args) for n in range(5)]
    assert dist_moms == wrap_moms, 'wrapped moments are different'

    # compare confidence intervals
    log.debug('... method `interval`')
    dist_intv = dist.interval(0.1, *dist_args)
    wrap_intv = wrap.interval(0.1, *wrap_args)
    assert dist_intv == wrap_intv, 'wrapped interval is different'

    # compare expectations
    log.debug('... method `expect`')
    # expect needs loc and scale split from args
    a, loc, scale = dist._parse_args(*dist_args)
    # function to take expectation of
    dist_expe = dist.expect(lambda x: x/np.sqrt(1 + x**2), a, loc, scale)
    wrap_expe = wrap.expect(lambda x: x/np.sqrt(1 + x**2), wrap_args)
    assert dist_expe == wrap_expe, 'wrapped expectation is different'


def test_wrap_continuous():
    # function to parametrise normal with mean and variance
    def mean_and_variance(mean, variance):
        loc = mean
        scale = np.sqrt(variance)
        return loc, scale

    # test with the normal distribution
    _test_wrap(scipy.stats.norm, mean_and_variance, (1.0, 0.25))


def test_wrap_discrete():
    # function to parametrise poisson by skewness
    def skewness(skew):
        mu = 1/skew**2
        return mu,

    # test with the Poisson distribution
    _test_wrap(scipy.stats.poisson, skewness, (0.9,))
