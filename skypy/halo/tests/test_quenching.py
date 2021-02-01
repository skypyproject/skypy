import numpy as np
import pytest
from scipy import stats
from collections import Counter

from skypy.halo.quenching import environment_quenched, mass_quenched


@pytest.mark.flaky
def test_environment_quenched():
    # Test the quenching process follows a binomial distribution
    n, p = 1000, 0.7
    quenched = environment_quenched(n, p)
    number_quenched = Counter(quenched)[True]

    p_value = stats.binom_test(number_quenched, n=n, p=p,
                               alternative='two-sided')
    assert p_value > 0.01


@pytest.mark.flaky
def test_mass_quenched():
    # Test the quenching process follows a binomial distribution if the
    # logarithmic halo masses are symmetrical about the offset
    n = 1000
    offset, width = 1.0e12, 0.5
    halo_mass = 10 ** np.random.uniform(11, 13, n)
    quenched = mass_quenched(halo_mass, offset, width)
    number_quenched = Counter(quenched)[True]

    p_value = stats.binom_test(number_quenched, n=n, p=0.5,
                               alternative='two-sided')
    assert p_value > 0.01
