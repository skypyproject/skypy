import numpy as np
import scipy.stats
import astropy.units as u
from astropy.tests.helper import raises
import logging

from skypy.stats import parametrise, examples


_WRAPPED_METHODS = [
    'median',
    'mean',
    'var',
    'std',
    'entropy',
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


def _test_wrap(dist, argsfn, wrap_args, units=None):
    logging.debug('testing wrapped %s' % str(dist))

    # wrap the distribution
    wrap = parametrise(dist, argsfn, units=units)

    # arguments for distribution
    dist_args = argsfn(*wrap_args)

    # return units of methods
    units_dict = {}
    if units:
        units_dict['median'] = units
        units_dict['mean'] = units
        units_dict['var'] = units**2
        units_dict['std'] = units
        units_dict['pdf'] = 1/units
        units_dict['ppf'] = units
        units_dict['isf'] = units

    # sample random number
    logging.debug('... method `rvs`')
    np.random.seed(1234)
    x = dist.rvs(*dist_args)
    logging.debug('    x = {}'.format(x))
    np.random.seed(1234)
    y = wrap.rvs(*wrap_args)
    logging.debug('    y = {}'.format(y))
    u = units or 1
    assert x*u == y, 'wrapped distribution sampled different value'

    # test plain methods
    for m in _WRAPPED_METHODS:
        if hasattr(dist, m):
            logging.debug('... method `%s`' % m)
            f = getattr(dist, m)
            g = getattr(wrap, m)
            a = f(*dist_args)
            logging.debug('    a = {}'.format(a))
            b = g(*wrap_args)
            logging.debug('    b = {}'.format(b))
            u = units_dict.get(m, 1)
            assert a*u == b, 'different result for wrapped method `%s`' % m

    # test methods of x or k
    for m in _WRAPPED_METHODS_OF_X:
        if hasattr(dist, m):
            logging.debug('... method `%s`' % m)
            f = getattr(dist, m)
            g = getattr(wrap, m)
            a = f(x, *dist_args)
            logging.debug('    a = {}'.format(a))
            b = g(y, *wrap_args)
            logging.debug('    b = {}'.format(b))
            u = units_dict.get(m, 1)
            assert a*u == b, 'different result for wrapped method `%s`' % m

    # test methods of quantile
    q = np.random.rand()
    for m in _WRAPPED_METHODS_OF_Q:
        if hasattr(dist, m):
            logging.debug('... method `%s`' % m)
            f = getattr(dist, m)
            g = getattr(wrap, m)
            a = f(q, *dist_args)
            logging.debug('    a = {}'.format(a))
            b = g(q, *wrap_args)
            logging.debug('    b = {}'.format(b))
            u = units_dict.get(m, 1)
            assert a*u == b, 'different result for wrapped method `%s`' % m

    # compare supports
    logging.debug('... method `support`')
    dist_supp = dist.support(*dist_args)
    logging.debug('    dist_supp = {}'.format(dist_supp))
    wrap_supp = wrap.support(*wrap_args)
    logging.debug('    wrap_supp = {}'.format(wrap_supp))
    if units:
        dist_supp = type(dist_supp)([x*units for x in dist_supp])
    assert dist_supp == wrap_supp, 'wrapped support is different'

    # compare stats
    logging.debug('... method `stats`')
    dist_mvsk = dist.stats(*dist_args, moments='mvsk')
    wrap_mvsk = wrap.stats(*wrap_args, moments='mvsk')
    assert dist_mvsk == wrap_mvsk, 'wrapped stats are different'

    # compare a number of moments
    logging.debug('... method `moment`')
    u = units or 1
    dist_moms = [dist.moment(n, *dist_args)*u**n for n in range(5)]
    logging.debug('    dist_moms = {}'.format(dist_moms))
    wrap_moms = [wrap.moment(n, *wrap_args) for n in range(5)]
    logging.debug('    wrap_moms = {}'.format(wrap_moms))
    assert dist_moms == wrap_moms, 'wrapped moments are different'

    # compare confidence intervals
    logging.debug('... method `interval`')
    dist_intv = dist.interval(0.1, *dist_args)
    logging.debug('    dist_intv = {}'.format(dist_intv))
    wrap_intv = wrap.interval(0.1, *wrap_args)
    logging.debug('    wrap_intv = {}'.format(wrap_intv))
    if units:
        dist_intv = type(dist_intv)([x*units for x in dist_intv])
    assert dist_intv == wrap_intv, 'wrapped interval is different'

    # compare expectations
    logging.debug('... method `expect`')
    # expect needs loc and scale split from args
    a, loc, scale = dist._parse_args(*dist_args)
    logging.debug('    a = {}, loc = {}, scale = {}'.format(a, loc, scale))
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


def test_wrap_units():
    logging.debug('testing wrapping with units')

    # test with truncexpon and units
    _test_wrap(scipy.stats.truncexpon, lambda x: (x,), (1.,), u.m)


@raises(TypeError)
def test_wrap_incorrect_units():
    logging.debug('testing wrapping with incorrect units')

    # test with incorrect units
    _test_wrap(scipy.stats.norm, lambda x: (x,), (1.,), units=3.)


def test_examples():
    # call without arguments
    @examples
    def examples_without_arguments():
        '''A function with a docstring.'''
    logging.debug(examples_without_arguments.__doc__)

    # call with empty arguments
    @examples()
    def examples_with_empty_arguments():
        '''A function with a docstring.'''
    logging.debug(examples_with_empty_arguments.__doc__)

    # call with explicit doc
    @examples(doc='A different docstring.')
    def examples_with_doc():
        '''A function with a docstring.'''
    logging.debug(examples_with_doc.__doc__)

    # call with explicit name
    @examples(name='a_different_name')
    def examples_with_name():
        '''A function with a docstring.'''
    logging.debug(examples_with_name.__doc__)

    # call with explicit shapes
    @examples(shapes='alpha, beta, gamma')
    def examples_with_shapes():
        '''A function with a docstring.'''
    logging.debug(examples_with_shapes.__doc__)

    # call with explicit shapes and args
    @examples(shapes='alpha, beta, gamma', args=(1., 2., 3.))
    def examples_with_shapes_and_args():
        '''A function with a docstring.'''
    logging.debug(examples_with_shapes_and_args.__doc__)

    # call with explicit module
    @examples(module='a_different_module')
    def examples_with_module():
        '''A function with a docstring.'''
    logging.debug(examples_with_module.__doc__)

    # call with implicit shapes for function
    @examples(args=(1., 2., 3.))
    def examples_with_function_shapes(alpha, beta, gamma):
        '''A function with a docstring.'''
    logging.debug(examples_with_function_shapes.__doc__)

    # call with implicit shapes for class
    @examples(args=(1., 2., 3.))
    class examples_with_class_shapes():
        '''A class with a docstring.'''
        def _pdf(self, x, alpha, beta, gamma):
            pass
    logging.debug(examples_with_class_shapes.__doc__)

    # call with implicit shapes from attribute
    @examples(args=(1., 2., 3.))
    class examples_with_attribute_shapes():
        '''A class with a docstring.'''
        def _pdf(self, x, *args):
            pass
        shapes = 'alpha, beta, gamma'
    logging.debug(examples_with_attribute_shapes.__doc__)
