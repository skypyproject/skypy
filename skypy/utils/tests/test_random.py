import numpy as np
import scipy.stats
from scipy import integrate
from skypy.utils import random, special
import scipy.special as sp


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
        a = special.gammaincc(alpha + 1, x_min)
        b = special.gammaincc(alpha + 1, x)
        c = special.gammaincc(alpha + 1, x_min)
        d = special.gammaincc(alpha + 1, x_max)
        return (a - b) / (c - d)

    sample = random.schechter(alpha, x_min=x_min, x_max=x_max,
                              resolution=100, size=1000)

    # Test the distribution of galaxy properties follows the right distribution
    p_value = scipy.stats.kstest(sample, calc_cdf)[1]
    assert p_value > 0.01


def test_conditional_prob_shmr():

    # Test the conditional_prob_shmr function against precomputed values
    x = np.linspace(11., 15., 20)
    cdf = random.conditional_prob_shmr(min(x), max(x), 20)
    sample = np.array([
             0., 0.05263158, 0.10526316, 0.15789474, 0.21052632,
             0.26315789, 0.31578947, 0.36842105, 0.42105263, 0.47368421,
             0.52631579, 0.57894737, 0.63157895, 0.68421053, 0.73684211,
             0.78947368, 0.84210526, 0.89473684, 0.94736842, 1.])

    np.testing.assert_allclose(cdf, sample)

    # Test the conditional_prob_shmr function sampling dimensionless x values
    x_min = 11.
    x_max = 15.

    def calc_cdf(x):
        x = np.linspace(x_min, x_max, 1000)
        cdf = integrate.cumtrapz(sp.erf(x), x, initial=0)
        cdf = cdf/cdf[-1]
        return cdf
    sample = random.conditional_prob_shmr(x_min=x_min, x_max=x_max, size=1000)

    # Test the distribution of galaxy properties follows the right distribution
    p_value = scipy.stats.kstest(sample, calc_cdf)[1]
    assert p_value > 0.01
