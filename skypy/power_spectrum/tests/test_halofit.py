import numpy as np
from astropy.cosmology import Planck15
from astropy.units import allclose
from astropy.utils.data import get_pkg_data_filename
import pytest
from skypy.power_spectrum import halofit_smith
from skypy.power_spectrum import halofit_takahashi
from skypy.power_spectrum import halofit_bird


# Power spectrum data for tests
linear_power_filename = get_pkg_data_filename('data/linear_power.txt')
truth_smith_filename = get_pkg_data_filename('data/truth_smith.txt')
truth_takahashi_filename = get_pkg_data_filename('data/truth_takahashi.txt')
truth_bird_filename = get_pkg_data_filename('data/truth_bird.txt')
linear_power = np.loadtxt(linear_power_filename)
truth_smith = np.loadtxt(truth_smith_filename)
truth_takahashi = np.loadtxt(truth_takahashi_filename)
truth_bird = np.loadtxt(truth_bird_filename)


def test_halofit():
    """Test Smith, Takahashi and Bird Halofit models with Planck15 cosmology"""

    # Wavenumbers and redshifts for tests
    k = np.logspace(-4, 2, 100, base=10)
    z = np.linspace(0, 2, 5)

    # Non-linear power spectra from Smith, Takahashi and Bird models
    nl_power_smith = halofit_smith(k, z, linear_power, Planck15)
    nl_power_takahashi = halofit_takahashi(k, z, linear_power, Planck15)
    nl_power_bird = halofit_bird(k, z, linear_power, Planck15)

    # Test shape of outputs
    assert np.shape(nl_power_smith) == np.shape(linear_power)
    assert np.shape(nl_power_takahashi) == np.shape(linear_power)
    assert np.shape(nl_power_bird) == np.shape(linear_power)

    # Test outputs against precomputed values
    assert allclose(nl_power_smith, truth_smith)
    assert allclose(nl_power_takahashi, truth_takahashi)
    assert allclose(nl_power_bird, truth_bird)

    # Test when redshift is a scalar
    z_scalar = z[0]
    power_1d = linear_power[0, :]
    truth_scalar_redshift = truth_smith[0, :]
    smith_scalar_redshift = halofit_smith(k, z_scalar, power_1d, Planck15)
    assert allclose(smith_scalar_redshift, truth_scalar_redshift)
    assert np.shape(smith_scalar_redshift) == np.shape(power_1d)

    # Test for failure when wavenumber is a scalar
    k_scalar = k[0]
    power_1d = linear_power[:, 0]
    with pytest.raises(TypeError):
        halofit_smith(k_scalar, z, power_1d, Planck15)

    # Test for failure when wavenumber array is the wrong size
    k_wrong_size = np.logspace(-4.0, 2.0, 7)
    with pytest.raises(ValueError):
        halofit_smith(k_wrong_size, z, linear_power, Planck15)

    # Test for failure when redshift array is the wrong size
    z_wrong_size = np.linspace(0.0, 2.0, 3)
    with pytest.raises(ValueError):
        halofit_smith(k, z_wrong_size, linear_power, Planck15)

    # Test for failure when wavenumber is negative
    k_negative = np.copy(k)
    k_negative[0] = -1.0
    with pytest.raises(ValueError):
        halofit_smith(k_negative, z, linear_power, Planck15)

    # Test for failure when wavenumber is zero
    k_zero = np.copy(k)
    k_zero[0] = 0.0
    with pytest.raises(ValueError):
        halofit_smith(k_zero, z, linear_power, Planck15)

    # Test for failure when redshift is negative
    z_negative = np.copy(z)
    z_negative[0] = -1.0
    with pytest.raises(ValueError):
        halofit_smith(k, z_negative, linear_power, Planck15)

    # Test for failure when linear power spectrum is negative
    power_negative = np.copy(linear_power)
    power_negative[0, 0] = -1.0
    with pytest.raises(ValueError):
        halofit_smith(k, z, power_negative, Planck15)

    # Test for failure when wavenumber array is not in asscending order
    k_wrong_order = k[::-1]
    with pytest.raises(ValueError):
        halofit_smith(k_wrong_order, z, linear_power, Planck15)
