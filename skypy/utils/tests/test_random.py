import numpy as np
import scipy.stats

import skypy.utils.random as random


def test_schechter_cdf():
    # Test that sampling corresponds to sampling from the right pdf.
    x = np.logspace(np.log10(0.1), np.log10(10), 10)
    cdf = random._schechter_cdf(x, min(x), max(x), -1.3)
    sample = np.array([0., 0.31938431, 0.57066007, 0.75733831, 0.88345969,
                       0.95631112, 0.98886695, 0.99846589, 0.99992254, 1.])
    np.testing.assert_allclose(cdf, sample)
