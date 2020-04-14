import numpy as np
import scipy.stats
import scipy.integrate
from skypy.utils import random, special


def test_schechter_cdf():
    # Test the _schechter_cdf function against precomputed values
    x = np.logspace(np.log10(0.1), np.log10(10), 10)
    cdf = random._schechter_cdf(x, min(x), max(x), -1.3)
    sample = np.array([0., 0.31938431, 0.57066007, 0.75733831, 0.88345969,
                       0.95631112, 0.98886695, 0.99846589, 0.99992254, 1.])

    np.testing.assert_allclose(cdf, sample)


def test_schechter():

    # Test the schechter function, sampling dimensionless x values
    alpha = -1.3
    x_min = 1e-10
    x_max = 1e2

    def calc_cdf(x):
        a = special.upper_incomplete_gamma(alpha + 1, x_min)
        b = special.upper_incomplete_gamma(alpha + 1, x)
        c = special.upper_incomplete_gamma(alpha + 1, x_min)
        d = special.upper_incomplete_gamma(alpha + 1, x_max)
        return (a - b) / (c - d)

    sample = random.schechter(alpha, x_min=x_min, x_max=x_max,
                              resolution=100, size=1000)

    # Test the distribution of galaxy properties follows the right distribution
    p_value = scipy.stats.kstest(sample, calc_cdf)[1]
    assert p_value > 0.01
