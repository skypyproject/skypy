import numpy as np
import scipy.stats
import scipy.integrate


import skypy.halo.mass as mass


def test_press_schechter():
    # Test the schechter function, sampling dimensionless x values
    n, m_star = 1, 1e9
    alpha = - 0.5 * (n + 9.0) / (n + 3.0)

    def calc_cdf(m):
        x = np.power(m/m_star, 1 + n/3)
        pdf = np.power(x, alpha) * np.exp(- x)
        cdf = scipy.integrate.cumtrapz(pdf, x, initial=0)
        cdf = cdf / cdf[-1]
        return cdf

    sample = mass.press_schechter(n, m_star, x_min=1e-1, x_max=1e2,
                                  resolution=100, size=1000)

    # Test the distribution of galaxy mass follows the right distribution
    p_value = scipy.stats.kstest(sample, calc_cdf)[1]
    assert p_value > 0.01
