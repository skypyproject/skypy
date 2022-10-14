import numpy as np
from scipy.stats import kstest
from scipy.special import gamma
from scipy.integrate import cumtrapz
from skypy.utils.random import schechter

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
    samples = schechter_vdf(alpha = alpha, beta = beta, vd_star = vd_star, 
                            vd_min = vd_min, vd_max = vd_max, size=size, resolution = resolution)

    # test output is within limits and size
    assert np.sum((vd_min <= samples) & (samples <= vd_max)) == size

    # test sampling against alternative implementation
    def vdf_func(x):
        return phi_star*(x/vd_star)**alpha*np.exp(-(x/vd_star)**beta)*(beta/gamma(alpha/beta))*(1/x)

    def calc_cdf(size):
        lnx = np.linspace(vd_min, vd_max, size)
        pdf = vdf_func(lnx)
        cdf = pdf
        np.cumsum((pdf[1:]+pdf[:-1])/2*np.diff(lnx), out=cdf[1:])
        cdf[0] = 0
        cdf /= cdf[-1]

        t_lower = np.interp(vd_min, lnx, cdf)
        t_upper = np.interp(vd_max, lnx, cdf)

        u = np.random.uniform(t_lower, t_upper, size=size)
        lnx_sample = np.interp(u, cdf, lnx)

        return lnx_sample

    p_value = kstest(samples, calc_cdf(size))
    assert p_value[1] > 0.01
    
