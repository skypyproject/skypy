import numpy as np
from astropy.cosmology import Planck15
from astropy.units import allclose
from skypy.power_spectrum import eisenstein_hu


import skypy.halos.mass as mass

# Precomputed values for the test, for a Planck15 cosmology at redshift 0 and a
# power spectrum given by the Eisenstein and Hu fitting formula
# Models: Press-Schechter and Sheth-Tormen
mass_array = [1.00000000e+10, 3.16227766e+10, 1.00000000e+11, 3.16227766e+11,
              1.00000000e+12, 3.16227766e+12, 1.00000000e+13, 3.16227766e+13,
              1.00000000e+14, 3.16227766e+14]
ST_fsigma = [0.21528702, 0.23193284, 0.26101991, 0.28439985, 0.29890766,
             0.31534398, 0.32025265, 0.31040398, 0.26278585, 0.20735536]
PS_fsigma = [0.24691981, 0.28079417, 0.34311288, 0.39593294, 0.4299622,
             0.47012612, 0.48356179, 0.4643191, 0.36519188, 0.25891939]
E_fsigma = [0.19994134, 0.21541551, 0.24248344, 0.26428697, 0.27785953,
            0.2933539, 0.2981575, 0.28948277, 0.24590202, 0.19472173]
ST_massf = [5.32103203e-12, 1.81481123e-12, 3.37138805e-13, 2.91376301e-14,
            2.16697446e-15, 3.36378038e-16, 2.56359970e-17, 3.61363924e-18,
            3.58933700e-19, 2.28424628e-20]
PS_massf = [6.10286780e-12, 2.19713785e-12, 4.43171821e-13, 4.05645349e-14,
            3.11707341e-15, 5.01484440e-16, 3.87087767e-17, 5.40547738e-18,
            4.98807967e-19, 2.85228053e-20]
E_massf = [4.94174854e-12, 1.68556767e-12, 3.13196716e-13, 2.70770044e-14,
           2.01438299e-15, 3.12921173e-16, 2.38672955e-17, 3.37008015e-18,
           3.35872438e-19, 2.14507308e-20]

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
