import numpy as np
import scipy.stats
import scipy.integrate
import pytest

import skypy.galaxy.luminosity as lum
import skypy.utils.special as special
import skypy.utils.astronomy as astro


def test_calculate_luminosity_star():
    luminosity_star = lum._calculate_luminosity_star(1, 2.5, 5)
    assert luminosity_star == 0.001

    redshift = np.array([0.1, 0.6, 1])
    luminosity_star = lum._calculate_luminosity_star(redshift, 2.5, 5)
    result = np.array([0.00794328, 0.00251189, 0.001])
    np.testing.assert_allclose(luminosity_star, result, rtol=1e-05)


def test_herbel_luminosities():
    # Test that error is returned if redshift input is an array but size !=
    # None and size != redshift,size
    with pytest.raises(ValueError):
        lum.herbel_luminosities(np.array([1, 2]), -1.3, -0.9408582,
                                -20.40492365, size=3)

    # Test that sampling corresponds to sampling from the right pdf.
    # For this, we sample an array of luminosities for redshift z = 1.0 and we
    # compare it to the corresponding cdf.

    def calc_pdf(luminosity, z, alpha, a_m, b_m, absolute_magnitude_max=-16.0,
                 absolute_magnitude_min=-28.0):
        luminosity_min = astro.luminosity_from_absolute_magnitude(
            absolute_magnitude_max)
        luminosity_max = astro.luminosity_from_absolute_magnitude(
            absolute_magnitude_min)
        lum_star = lum._calculate_luminosity_star(z, a_m, b_m)
        c = special.upper_incomplete_gamma(alpha + 1,
                                           luminosity_min / lum_star)
        d = special.upper_incomplete_gamma(alpha + 1,
                                           luminosity_max / lum_star)
        return 1. / lum_star * np.power(luminosity / lum_star, alpha) * np.exp(
            - luminosity / lum_star) / (c - d)

    def calc_cdf(L):
        a_m = -0.9408582
        b_m = -20.40492365
        alpha = -1.3
        pdf = calc_pdf(L, 1.0, alpha, a_m, b_m)
        cdf = scipy.integrate.cumtrapz(pdf, L, initial=0)
        cdf = cdf / cdf[-1]
        return cdf

    sample = lum.herbel_luminosities(1.0, -1.3, -0.9408582, -20.40492365,
                                     size=1000)
    p_value = scipy.stats.kstest(sample, calc_cdf)[1]
    assert p_value >= 0.01


def test_exponential_distribution():
    # When alpha=0, L*=1 and x_min~0 we get a truncated exponential
    q_max = 1e2
    sample = lum.herbel_luminosities(0, 0, 0, 0, size=1000,
                                     x_min=1e-10, x_max=q_max,
                                     resolution=1000)
    d, p_value = scipy.stats.kstest(sample, 'truncexpon', args=(q_max,))
    assert p_value >= 0.01
