import numpy as np
from astropy.cosmology import Planck15
from astropy.units import allclose
from skypy.power_spectrum import eisenstein_hu


import skypy.halo.mass as mass

# Precomputed values for the test, for a Planck15 cosmology at redshift 0 and a
# power spectrum given by the Eisenstein and Hu fitting formula
# Models: Press-Schechter and Sheth-Tormen
mass_array = [1.00000000e+10, 3.16227766e+10, 1.00000000e+11, 3.16227766e+11,
              1.00000000e+12, 3.16227766e+12, 1.00000000e+13, 3.16227766e+13,
              1.00000000e+14, 3.16227766e+14]
ST_fsigma = [0.22851856, 0.24659885, 0.27788, 0.30138876, 0.31338251,
             0.32004239, 0.31068427, 0.2676586, 0.16714694, 0.0957242]
PS_fsigma = [0.27373595, 0.31173523, 0.38096078, 0.4358908, 0.4651783,
             0.48375949, 0.46491057, 0.37505525, 0.18939675, 0.08453155]
E_fsigma = [0.21224081, 0.22905786, 0.25820042, 0.28018667, 0.29148879,
            0.29805329, 0.28973637, 0.25038455, 0.1574158, 0.09077113]
ST_massf = [5.91918389e-12, 2.04006848e-12, 3.90033154e-13, 3.43105199e-14,
            2.47935792e-15, 3.85619829e-16, 2.77447365e-17, 3.53176402e-18,
            2.69047690e-19, 1.13766655e-20]
PS_massf = [7.09042387e-12, 2.57893025e-12, 5.34717624e-13, 4.96224201e-14,
            3.68030594e-15, 5.82882944e-16, 4.15174583e-17, 4.94886632e-18,
            3.04862034e-19, 1.00464376e-20]
E_massf = [5.49755090e-12, 1.89495500e-12, 3.62410841e-13, 3.18968432e-14,
           2.30614353e-15, 3.59125102e-16, 2.58740463e-17, 3.30383237e-18,
           2.53383973e-19, 1.07880011e-20]

cosmo = Planck15
k = np.logspace(-3, 1, num=10, base=10.0)
A_s, n_s = 2.1982e-09, 0.969453
Pk = eisenstein_hu(k, A_s, n_s, cosmo, kwmap=0.02, wiggle=True)


def test_halo_mass_function():
    # Test the output and shape is correct given an array of masses
    m_array = np.asarray(mass_array)

    # Any particular ellipsoidal collapse model
    params_E = (0.3, 0.7, 0.3, 1.686)
    fE = mass.ellipsoidal_collapse_function
    array_output_E = mass.halo_mass_function(m_array, k, Pk, 1.0, cosmo,
                                             fE, params=params_E)
    assert array_output_E.shape == m_array.shape
    assert allclose(array_output_E, E_massf)

    # Sheth and Tormen collapse model
    array_output_ST = mass.sheth_tormen_mass_function(m_array, k, Pk,
                                                      1.0, cosmo)
    assert array_output_ST.shape == m_array.shape
    assert allclose(array_output_ST, ST_massf)

    # Press-Schechter collapse model
    array_output_PS = mass.press_schechter_mass_function(m_array, k, Pk,
                                                         1.0, cosmo)
    assert array_output_PS.shape == m_array.shape
    assert allclose(array_output_PS, PS_massf)


def test_halo_mass_sampler():
    # Test the output shape is correct given the sample size
    n_samples = 1000
    m_min, m_max, resolution = 10**9, 10**12, 100
    # Any particular ellipsoidal collapse model
    params_E = (0.3, 0.7, 0.3, 1.686)
    fE = mass.ellipsoidal_collapse_function
    array_output_E = mass.halo_mass_sampler(m_min, m_max, resolution, k, Pk,
                                            1.0, cosmo, fE, params=params_E,
                                            size=n_samples)
    assert len(array_output_E) == n_samples

    # Sheth and Tormen collapse model
    array_output_PS = mass.sheth_tormen(10**9, 10**12, 100, k,
                                        Pk, 1.0, cosmo, size=n_samples)

    assert len(array_output_PS) == n_samples

    # Press-Schechter collapse model
    array_output_PS = mass.press_schechter(10**9, 10**12, 100, k,
                                           Pk, 1.0, cosmo, size=n_samples)

    assert len(array_output_PS) == n_samples


def test_ellipsoidal_collapse_function():
    # Test any ellipsoidal model against precomputed values
    m_array = np.asarray(mass_array)
    sigma = np.sqrt(mass._sigma_squared(m_array, k, Pk, 1.0, cosmo))
    params_E = (0.3, 0.7, 0.3, 1.686)
    fE = mass.ellipsoidal_collapse_function(sigma, params_E)
    assert allclose(fE, E_fsigma)

    # Test the Sheth and Tormen model against precomputed values
    m_array = np.asarray(mass_array)
    sigma = np.sqrt(mass._sigma_squared(m_array, k, Pk, 1.0, cosmo))
    fst = mass.sheth_tormen_collapse_function(sigma)
    assert allclose(fst, ST_fsigma)

    # Test the Press-Schechter model against precomputed values
    m_array = np.asarray(mass_array)
    sigma = np.sqrt(mass._sigma_squared(m_array, k, Pk, 1.0, cosmo))
    fps = mass.press_schechter_collapse_function(sigma)
    assert allclose(fps, PS_fsigma)


def test_number_subhalos():
    # Test analytic solution for the mean number of subhalos
    halo_mass = 1.0e12
    shm_min = halo_mass / 100
    alpha, beta, gamma_M, x = 0.0, 0.39, 0.18, 3.0
    nsh_output = mass.number_subhalos(halo_mass, alpha, beta, gamma_M, x, shm_min, noise=False)
    nsh_mean = (gamma_M / beta) * np.exp(- shm_min / (x * beta * halo_mass))

    assert round(nsh_output) == round(nsh_mean)

    # Test for array input of halo parents
    halo_parents = np.array([1.0e12, 1.0e14])
    shm_min = 1.0e6
    alpha, beta, gamma_M, x = 1.9, 1.0, 0.3, 1.0
    array_nsh = mass.number_subhalos(halo_parents, alpha, beta, gamma_M, x, shm_min)

    assert len(array_nsh) == len(halo_parents)


def test_subhalo_mass_sampler():
    # Test the output shape is correct given the sample size
    halo_mass, shm_min = 1.0e12, 1.0e6
    alpha, beta, x = 1.9, 1.0, 1.0
    nsh = 20
    array_output = mass.subhalo_mass_sampler(halo_mass, nsh, alpha, beta, x, shm_min, 100)

    assert len(array_output) == np.sum(nsh)

    # For each halo test that each subhalo satisfy shm_min < m < shm_max
    shm_max = 0.5 * halo_mass

    assert np.all(array_output > shm_min) and np.all(array_output < shm_max)

    # Repeat the tests for arrays of halos
    halo_mass = np.array([1.0e12, 1.0e14])
    nsh = np.array([10, 100])
    shm_max = 0.5 * halo_mass
    array_output = mass.subhalo_mass_sampler(halo_mass, nsh, alpha, beta, x, shm_min, 100)

    assert len(array_output) == np.sum(nsh)
    assert np.all(array_output[:10] > shm_min) and np.all(array_output[:10] < 0.5 * halo_mass[0])
    assert np.all(array_output[10:] > shm_min) and np.all(array_output[10:] < 0.5 * halo_mass[1])
