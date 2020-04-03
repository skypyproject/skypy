import numpy as np
import numpy.testing as npt
import scipy.stats as st
import scipy.stats.tests as stt


# list of exported checks
__all__ = [
    'check_internals',
    'check_normalisation',
    'check_functions',
    'check_moments',
    'check_sample_size',
    'check_sample_distribution'
]


def check_internals(rv, args):
    '''check internal settings of rv'''

    # check internal properties
    stt.common_tests.check_edge_support(rv, args)
    stt.common_tests.check_random_state_property(rv, args)
    stt.common_tests.check_pickling(rv, args)


def check_normalisation(rv, args):
    '''check normalisation of rv'''

    # check that PDF is normalised
    stt.common_tests.check_normalization(rv, args, rv.name)


def check_functions(rv, args, size=10):
    '''check functions (CDF, PPF, SF, ISF, ...) of rv'''

    # test points
    x = rv.rvs(*args, size=size)

    # quantiles of test points
    q = rv.cdf(x, *args)

    # check CDF and SF
    npt.assert_allclose(q + rv.sf(x, *args), 1.)

    # check CDF and PPF
    npt.assert_allclose(rv.ppf(q, *args), x)

    # check SF and ISF
    npt.assert_allclose(rv.isf(1-q, *args), x)


def check_moments(rv, args):
    '''check moments of rv'''

    # get moments from rv
    m, v, s, k = rv.stats(*args, moments='mvsk')

    # check moments
    stt.common_tests.check_mean_expect(rv, args, m, rv.name)
    stt.common_tests.check_var_expect(rv, args, m, v, rv.name)
    stt.common_tests.check_skew_expect(rv, args, m, v, s, rv.name)
    stt.common_tests.check_kurt_expect(rv, args, m, v, k, rv.name)
    stt.common_tests.check_moment(rv, args, m, v, rv.name)


def check_sample_size(rv, args):
    '''check the sample size of rv'''

    # make sure args are a tuple
    assert np.ndim(args) == 1, 'args must be tuple'

    # args must not be multidimensional
    assert np.broadcast(*args).ndim <= 1, 'dimensional args detected'

    # sample a single value
    rvs = rv.rvs(*args)
    assert np.isscalar(rvs), 'sampling without size did not produce scalar'

    # sample 1d array
    rvs = rv.rvs(*args, size=10)
    assert rvs.shape == (10,), 'sampling did not produce 1d array'

    # sample 2d array
    rvs = rv.rvs(*args, size=(10, 5))
    assert rvs.shape == (10, 5), 'sampling did not produce 2d array'

    # sample with implicit sizes
    ndargs = [np.tile(a, (2,)*(i+1)) for i, a in enumerate(args)]
    rvs = rv.rvs(*ndargs)
    assert rvs.shape == np.shape(ndargs[-1]), \
        'sampling did not produce array with implicit size'


def check_sample_distribution(rv, args, equiv=None):
    '''check the sample distribution of rv'''

    # check sampling against own CDF
    rvs = rv.rvs(*args, size=1000)
    _, p = st.kstest(rvs, rv.cdf, args=args)
    assert p > 0.01, 'not distributed according to CDF, p = {}'.format(p)

    # check against equivalent rv
    if equiv is not None:
        for args in equiv:
            other_dist, other_args = equiv[args]
            rvs = rv.rvs(*args, size=1000)
            _, p = st.kstest(rvs, other_dist, args=other_args)
            assert p > 0.01, \
                '{}{} not distributed as {}{}, p = {}'.format(
                    rv.name, args, str(other_dist), other_args, p)


def check_rv(rv, args, equiv=None):
    '''standard checks for rv'''

    check_internals(rv, args)
    check_normalisation(rv, args)
    check_functions(rv, args)
    if rv._stats_has_moments:
        check_moments(rv, args)
    check_sample_size(rv, args)
    check_sample_distribution(rv, args, equiv)
