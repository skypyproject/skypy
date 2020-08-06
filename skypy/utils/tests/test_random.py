import numpy as np

from scipy.stats import kstest


def test_schechter():

    from skypy.utils.random import schechter

    def schechter_cdf_gen(alpha, x_min, x_max, resolution=100000):
        lnx = np.linspace(np.log(x_min), np.log(x_max), resolution)

        cdf = np.exp((alpha + 1)*lnx - np.exp(lnx))
        np.cumsum((cdf[:-1] + cdf[1:])/2*np.diff(lnx), out=cdf[1:])
        cdf[0] = 0
        cdf /= cdf[-1]

        return lambda x: np.interp(np.log(x), lnx, cdf)

    # Test the schechter function, sampling dimensionless x values
    alpha = -1.3
    x_min = 1e-10
    x_max = 1e2

    x = schechter(alpha, x_min=x_min, x_max=x_max, size=1000)

    assert np.all(x >= x_min)
    assert np.all(x <= x_max)

    # Test the distribution of galaxy properties follows the right distribution
    cdf = schechter_cdf_gen(alpha, x_min, x_max)
    _, p = kstest(x, cdf)
    assert p > 0.01


def test_schechter_gamma():

    from skypy.utils.random import schechter

    # when alpha > -1, x_min ≈ 0, x_max ≈ ∞, distribution is gamma
    alpha = np.random.uniform(-1, 2)
    x_min = 1e-10
    x_max = 1e+10

    x = schechter(alpha, x_min, x_max, resolution=100000, size=1000)

    _, p = kstest(x, 'gamma', args=(alpha+1,))
    assert p > 0.01
