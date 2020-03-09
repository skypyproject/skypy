import numpy as np
import scipy.stats
import scipy.integrate
import pytest

import skypy.galaxy.luminosities_herbel as lum
import skypy.utils.special as special
import skypy.utils.astronomy as astro


def test_cdf():
    q = np.logspace(np.log10(0.1), np.log10(10), 10)
    cdf = lum._cdf(q, min(q), max(q), -1.3)
    result = np.array([0., 0.31938431, 0.57066007, 0.75733831, 0.88345969,
                       0.95631112, 0.98886695, 0.99846589, 0.99992254, 1.])
    np.testing.assert_allclose(cdf, result)


def test_calculate_luminosity_star():
    luminosity_star = lum._calculate_luminosity_star(1, 2.5, 5)
    assert luminosity_star == 0.001

    redshift = np.array([0.1, 0.6, 1])
    luminosity_star = lum._calculate_luminosity_star(redshift, 2.5, 5)
    result = np.array([0.00794328, 0.00251189, 0.001])
    np.testing.assert_allclose(luminosity_star, result, rtol=1e-05)


def test_herbel_luminosities():
    # test that error is returned if redshift input is an array but size is not
    # None
    with pytest.raises(ValueError, match="If 'redshift' is an array,"
                                         " 'size' has to be None"):
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
