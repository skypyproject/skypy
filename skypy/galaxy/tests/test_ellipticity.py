import numpy as np
from scipy import stats


def test_cosmos235beta():
    from skypy.galaxy.ellipticity import cosmos235beta

    # check some properties
    assert np.isclose(cosmos235beta.mean(), 0.29930648342761523)
    assert np.isclose(cosmos235beta.var(), 0.032620322624451185)

    # sample a single ellipticity
    rvs = cosmos235beta.rvs()
    assert np.isscalar(rvs)

    # sample 10 ellipticities
    rvs = cosmos235beta.rvs(size=10)
    assert rvs.shape == (10,)

    # check sampling against own CDF
    D, p = stats.kstest(cosmos235beta.rvs, cosmos235beta.cdf, N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # check sampling against beta distribution with fitted parameters
    D, p = stats.kstest(cosmos235beta.rvs(size=1000),
                        stats.beta.cdf, args=(1.62499, 3.80420))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)


def test_cosmos252beta():
    from skypy.galaxy.ellipticity import cosmos252beta

    # check some properties
    assert np.isclose(cosmos252beta.mean(), 0.35321134013153520)
    assert np.isclose(cosmos252beta.var(), 0.035924585262542295)

    # sample a single ellipticity
    rvs = cosmos252beta.rvs()
    assert np.isscalar(rvs)

    # sample 10 ellipticities
    rvs = cosmos252beta.rvs(size=10)
    assert rvs.shape == (10,)

    # check sampling against own CDF
    D, p = stats.kstest(cosmos252beta.rvs, cosmos252beta.cdf, N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # check sampling against beta distribution with fitted parameters
    D, p = stats.kstest(cosmos252beta.rvs(size=1000),
                        stats.beta.cdf, args=(1.89294, 3.46630))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
