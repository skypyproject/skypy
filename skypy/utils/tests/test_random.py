import numpy as np

from scipy.stats import kstest


def test_schechter():

    from skypy.utils.random import schechter

    def schechter_cdf_gen(alpha, x_min, x_max):
        a = special.gammaincc(alpha + 1, x_min)
        b = special.gammaincc(alpha + 1, x_max)
        return lambda x: (a - special.gammaincc(alpha + 1, x)) / (a - b)

    # Test the schechter function, sampling dimensionless x values
    alpha = -1.3
    x_min = 1e-10
    x_max = 1e2

    x = schechter(alpha, x_min, x_max, size=1000)

    assert np.all(x >= x_min)
    assert np.all(x <= x_max)

    # Test the distribution of galaxy properties follows the right distribution
    cdf = schechter_cdf_gen(alpha, x_min, x_max)
    _, p = kstest(x, cdf)
    assert p > 0.01

    # Test output shape when scale is a scalar
    scale = 5
    samples = schechter(alpha, x_min, x_max, scale=scale)
    assert np.shape(samples) == np.shape(scale)

    # Test output shape when scale is an array
    scale = np.random.uniform(size=10)
    samples = schechter(alpha, x_min, x_max, scale=scale)
    assert np.shape(samples) == np.shape(scale)


def test_schechter_gamma():

    from skypy.utils.random import schechter

    # when alpha > -1, x_min ≈ 0, x_max ≈ ∞, distribution is gamma
    alpha = np.random.uniform(-1, 2)
    x_min = 1e-10
    x_max = 1e+10

    scale = 5
    x = schechter(alpha, x_min, x_max, resolution=100000, size=1000, scale=scale)

    _, p = kstest(x, 'gamma', args=(alpha+1, 0, scale))
    assert p > 0.01
