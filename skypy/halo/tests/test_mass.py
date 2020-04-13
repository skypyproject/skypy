# import numpy as np
import scipy.stats
# import pytest

import skypy.halo.mass as mass

# def test_press_schechter():

# Test that sampling corresponds to sampling from the right pdf.
# For this, we sample an array of luminosities for redshift z = 1.0 and we
# compare it to the corresponding cdf.

# sample = mass.press_schechter(1.0, 1e9, size=1000)
# p_value = scipy.stats.kstest(sample, calc_cdf)[1]
# assert p_value >= 0.01


def test_exponential_distribution():
    # When alpha=0,  and x_min~0 we get a truncated exponential
    q_max = 1e2
    sample = mass.press_schechter(-9, 1e9, size=1000,
                                  x_min=1e-10, x_max=q_max,
                                  resolution=1000)
    d, p_value = scipy.stats.kstest(sample, 'truncexpon', args=(q_max,))
    assert p_value >= 0.01
