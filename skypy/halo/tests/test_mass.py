import numpy as np
from astropy.cosmology import Planck15
from astropy.units import allclose
from skypy.power_spectrum import _eisenstein_hu as eh


import skypy.halo.mass as mass

# Precomputed values for the test, for a Planck15 cosmology at redshift 0 and a
# power spectrum given by the Eisenstein and Hu fitting formula
# Models: Press-Schechter and Sheth-Tormen
mass_array = [1.00000000e+10, 3.16227766e+10, 1.00000000e+11, 3.16227766e+11,
              1.00000000e+12, 3.16227766e+12, 1.00000000e+13, 3.16227766e+13,
              1.00000000e+14, 3.16227766e+14]
PS_fsigma = [0.30839245, 0.3469052, 0.38721761, 0.42703783, 0.46176454,
             0.48255685, 0.47362787, 0.41200955, 0.28215351, 0.11852313]
ST_fsigma = [0.24531776, 0.26301646, 0.28088396, 0.29791293, 0.31217744,
             0.32001412, 0.31466711, 0.28511577, 0.21918841, 0.1200341]
mass_funct = [3.35678772e-11, 3.87001910e-12, 4.46652718e-13, 5.15069390e-14,
              5.90581307e-15, 6.65419937e-16, 7.14923865e-17, 6.87084218e-18,
              5.22044135e-19, 2.43974103e-20]

cosmo = Planck15
k = np.logspace(-3, 1, num=10, base=10.0)
A_s, n_s = 2.1982e-09, 0.969453
Pk = eh.eisenstein_hu(k, A_s, n_s, cosmo, kwmap=0.02, wiggle=True)


def test_halo_mass_function():
    # Test the output shape is correct given an array of masses
    m_array = np.asarray(mass_array)
    # Sheth and Tormen collapse model
    param_ST = (0.3222, 0.707, 0.3, 1.686)
    fST = mass.sheth_tormen_collapse_function
    array_output_ST = mass.halo_mass_function(m_array, k, Pk, 0, cosmo,
                                              fST, params=param_ST)
    assert array_output_ST.shape == m_array.shape
    # Press-Schechter collapse model
    array_output_PS = mass.press_schechter_mass_function(m_array, k, Pk,
                                                         0, cosmo)
    assert array_output_PS.shape == m_array.shape


def halo_mass_sampler():
    # Test the output shape is correct given the sample size
    n_samples = 1000
    m_min, m_max, resolution = 10**9, 10**12, 100
    # Sheth and Tormen collapse model
    param_ST = (mass.sheth_tormen_collapse_function,
                (0.3222, 0.707, 0.3, 1.686))

    array_output_ST = mass.halo_mass_sampler(m_min, m_max, resolution, k, Pk,
                                             0, cosmo, mass.halo_mass_function,
                                             params=param_ST, size=n_samples)
    assert len(array_output_ST) == n_samples

    # Press-Schechter collapse model
    array_output_PS = mass.press_schechter_mass_sampler(10**9, 10**12, 100, k,
                                                        Pk, 0, cosmo)
    assert len(array_output_PS) == n_samples


def test_sheth_tormen_collapse_function():
    # Test against precomputed values
    m_array = np.asarray(mass_array)
    sigma = np.sqrt(mass._sigma_squared(m_array, k, Pk, 0, cosmo))
    ST_params = (0.3222, 0.707, 0.3, 1.686)
    fst = mass.sheth_tormen_collapse_function(sigma, params=ST_params)
    assert allclose(fst, ST_fsigma)


def test_press_schechter_collapse_function():
    # Test against precomputed values
    m_array = np.asarray(mass_array)
    sigma = np.sqrt(mass._sigma_squared(m_array, k, Pk, 0, cosmo))
    fps = mass.press_schechter_collapse_function(sigma)
    assert allclose(fps, PS_fsigma)
