import numpy as np
import pytest
from scipy import stats


@pytest.mark.flaky
def test_beta_ellipticity():

    from skypy.galaxy.ellipticity import beta_ellipticity

    # randomised ellipticity distribution with beta distribution parameters a,b
    # and the equivalent reparametrisation
    a, b = np.random.lognormal(size=2)
    e_ratio, e_sum = (a / (a + b), a + b)

    # Test scalar output
    assert np.isscalar(beta_ellipticity(e_ratio, e_sum))

    # Test array output
    assert beta_ellipticity(e_ratio, e_sum, size=10).shape == (10,)

    # Test broadcast output
    e_ratio2 = 0.5 * np.ones((13, 1, 5))
    e_sum2 = 0.5 * np.ones((7, 5))
    rvs = beta_ellipticity(e_ratio2, e_sum2)
    assert rvs.shape == np.broadcast(e_ratio2, e_sum2).shape

    # Kolmogorov-Smirnov test comparing ellipticity and beta distributions
    D, p = stats.kstest(beta_ellipticity(e_ratio, e_sum, size=1000), 'beta',
                        args=(a, b))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # Kolmogorov-Smirnov test comparing ellipticity and uniform distributions
    D, p = stats.kstest(beta_ellipticity(0.5, 2.0, size=1000), 'uniform')
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # Kolmogorov-Smirnov test comparing ellipticity and arcsine distributions
    D, p = stats.kstest(beta_ellipticity(0.5, 1.0, size=1000), 'arcsine')
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)


def test_ryden04():
    from skypy.galaxy.ellipticity import ryden04

    # sample a single ellipticity
    e = ryden04(0.222, 0.056, -1.85, 0.89)
    assert np.isscalar(e)

    # sample many ellipticities
    e = ryden04(0.222, 0.056, -1.85, 0.89, size=1000)
    assert np.shape(e) == (1000,)

    # sample with explicit shape
    e = ryden04(0.222, 0.056, -1.85, 0.89, size=(10, 10))
    assert np.shape(e) == (10, 10)

    # sample with implicit size
    e1 = ryden04([0.222, 0.333], 0.056, -1.85, 0.89)
    e2 = ryden04(0.222, [0.056, 0.067], -1.85, 0.89)
    e3 = ryden04(0.222, 0.056, [-1.85, -2.85], 0.89)
    e4 = ryden04(0.222, 0.056, -1.85, [0.89, 1.001])
    assert np.shape(e1) == np.shape(e2) == np.shape(e3) == np.shape(e4) == (2,)

    # sample with broadcasting rule
    e = ryden04([[0.2, 0.3], [0.4, 0.5]], 0.1, [-1.9, -2.9], 0.9)
    assert np.shape(e) == (2, 2)

    # sample with random parameters and check that result is in unit range
    args = np.random.rand(4)*[1., .1, -2., 1.]
    e = ryden04(*args, size=1000)
    assert np.all((e >= 0.) & (e <= 1.))

    # sample a spherical distribution
    e = ryden04(1-1e-99, 1e-99, -1e99, 1e-99, size=1000)
    assert np.allclose(e, 0.)
