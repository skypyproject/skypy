import numpy as np
import scipy.stats
import scipy.integrate

import skypy.utils.random as random


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

    def calc_cdf(x):
        pdf = np.power(x, alpha) * np.exp(- x)
        cdf = scipy.integrate.cumtrapz(pdf, x, initial=0)
        cdf = cdf / cdf[-1]
        return cdf

    sample = random.schechter(alpha, x_min=1e-10, x_max=1e2,
                              resolution=100, size=1000)

    # Test the distribution of galaxy properties follows the right distribution
    p_value = scipy.stats.kstest(sample, calc_cdf)[1]
    assert p_value > 0.01
