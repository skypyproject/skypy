from astropy.cosmology import default_cosmology
import numpy as np
from skypy.halo.abundance_matching import vale_ostriker


def test_vale_ostriker():
    """ Test Vale & Ostriker abundance matching algorithm"""

    cosmology = default_cosmology.get()
    k = np.logspace(-4, 2, 20)
    Pk = np.array([7.0554997e+02, 1.4269495e+03, 2.8806238e+03, 5.7748426e+03,
                   1.1311605e+04, 2.0794882e+04, 3.3334904e+04, 4.1292112e+04,
                   3.0335636e+04, 1.6623714e+04, 5.9353726e+03, 1.5235534e+03,
                   3.3850042e+02, 6.5466128e+01, 1.1470471e+01, 1.8625402e+00,
                   2.8532746e-01, 4.1803753e-02, 5.9166346e-03, 8.1485945e-04])

    halo_kwargs = {'m_min': 1.0E+9,
                   'm_max': 1.0E+12,
                   'resolution': 1000,
                   'size': 100,
                   'wavenumber': k,
                   'power_spectrum': Pk,
                   'growth_function': 0.40368249700456954,
                   'cosmology': cosmology, }
    subhalo_kwargs = {'alpha': -1.91,
                      'beta': 0.39,
                      'gamma_M': 0.18,
                      'x': 3,
                      'm_min': 1.0E+9,
                      'noise': True, }
    galaxy_kwargs = {'redshift': 1,
                     'M_star': -21.07994198,
                     'alpha': -0.5,
                     'm_lim': 35,
                     'size': 100, }

    mass, group, parent, mag = vale_ostriker(halo_kwargs, subhalo_kwargs, galaxy_kwargs)

    # Check monotonic mass-magnitude relation
    np.testing.assert_array_equal(np.argsort(mass)[::-1], np.argsort(mag))

    # Check group indexing
    assert np.min(group) == 0
    assert np.max(group) == np.unique(group).size - 1

    # Check minimum and maximum mass of parent halos
    np.testing.assert_array_less(halo_kwargs['m_min'], mass[parent])
    np.testing.assert_array_less(mass[parent], halo_kwargs['m_max'])

    # Check minimum and maximum mass of child halos
    for m, g in zip(mass[parent], group[parent]):
        children = np.logical_and(group == g, ~parent)
        np.testing.assert_array_less(subhalo_kwargs['m_min'], mass[children])
        np.testing.assert_array_less(mass[children], 0.5*m)

    # Check magnitude limit
    np.testing.assert_array_less(mag, galaxy_kwargs['m_lim'])
