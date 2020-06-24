import numpy as np
from scipy import stats
from collections import Counter


import skypy.halo.quenching as q


def test_environment_quenching():
    # Test the environment quenching function for 1000 subhalos
    number_subhalos = 1000
    quenched = q.environment_quenching(number_subhalos)

    # number of quenched subhalos
    number_quenched = Counter(quenched)[0]

    # Test the quenching process follows a binomial distribution
    p_value = stats.binom_test(number_quenched, n=number_subhalos, p=0.5,
                               alternative='greater')
    assert p_value > 0.05


def test_mass_quenching():
    # Test the mass quenching function for 1000 halos
    number_halos = 1000
    offset, width = 12, 6
    halo_mass = np.linspace(0, 24, num=number_halos)
    quenched = q.mass_quenching(halo_mass, offset, width)

    # number of quenched halos
    number_quenched = Counter(quenched)[0]

    # Test the quenching process follows a binomial distribution
    p_value = stats.binom_test(number_quenched, n=number_halos, p=0.5,
                               alternative='greater')
    assert p_value > 0.05
