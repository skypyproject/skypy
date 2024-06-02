import numpy as np
from scipy.stats import kstest
from scipy.special import gamma
from scipy.integrate import cumulative_trapezoid


def test_schechter_vdf():
    from skypy.galaxies.velocity_dispersion import schechter_vdf

    # test schechter velocity dispersion function
    vd_min = 1
    vd_max = 300
    size = 5000
    alpha = 2.32
    beta = 2.67
    vd_star = 161
    resolution = 3000
    phi_star = 8e-3
    samples = schechter_vdf(alpha=alpha, beta=beta, vd_star=vd_star,
                            vd_min=vd_min, vd_max=vd_max, size=size, resolution=resolution)

    # test output is within limits and size
    assert len(samples) == size
    assert vd_min <= samples.all() <= vd_max

    # test that sampling corresponds to sampling from the right pdf
    def calc_pdf(vd):
        return phi_star*(vd/vd_star)**alpha*np.exp(-(vd/vd_star)**beta) * \
            (beta/gamma(alpha/beta))*(1/vd)

    def calc_cdf(m):
        pdf = calc_pdf(m)
        cdf = cumulative_trapezoid(pdf, m, initial=0)
        cdf /= cdf[-1]
        return cdf

    p_value = kstest(samples, calc_cdf)[1]
    assert p_value > 0.01
