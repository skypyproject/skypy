import numpy as np
from scipy import stats


def test_cosmos235beta():
    from skypy.galaxy.ellipticity import cosmos235beta

    # check some properties
    assert np.isclose(cosmos235beta.mean(), 0.3001816324291912)
    assert np.isclose(cosmos235beta.var(), 0.03392735434342255)

    # sample a single ellipticity
    rvs = cosmos235beta.rvs()
    assert np.isscalar(rvs)

    # sample 10 ellipticities
    rvs = cosmos235beta.rvs(size=10)
    assert rvs.shape == (10,)

    # check sampling against beta distribution with fitted parameters
    D, p = stats.kstest(cosmos235beta.rvs(size=1000),
                        stats.beta.cdf, args=(1.55849386, 3.63334232))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)


def test_cosmos252beta():
    from skypy.galaxy.ellipticity import cosmos252beta

    # check some properties
    assert np.isclose(cosmos252beta.mean(), 0.35881953763575203)
    assert np.isclose(cosmos252beta.var(), 0.037323021185001096)

    # sample a single ellipticity
    rvs = cosmos252beta.rvs()
    assert np.isscalar(rvs)

    # sample 10 ellipticities
    rvs = cosmos252beta.rvs(size=10)
    assert rvs.shape == (10,)

    # check sampling against beta distribution with fitted parameters
    D, p = stats.kstest(cosmos252beta.rvs(size=1000),
                        stats.beta.cdf, args=(1.85303037, 3.31121008))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
