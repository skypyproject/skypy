import numpy as np
from astropy.cosmology import Planck15
from astropy.units import allclose
import pytest
from skypy.power_spectrum import halofit_smith
from skypy.power_spectrum import halofit_takahashi
from skypy.power_spectrum import halofit_bird


def test_halofit():
    """Test Smith, Takahashi and Bird Halofit models with Planck15 cosmology"""

    # Data for tests
    k = np.logspace(-4.0, 0.0, 5)
    z = np.linspace(0.0, 1.0, 2)
    p = [[705.54997046, 262.14967329], [6474.60158058, 2405.66190924],
         [37161.00990355, 13807.30920991], [9657.02613688, 3588.10339832],
         [114.60445565, 42.58170486]]
    ts = [[705.49027968, 262.13980368], [6469.19607307, 2404.75754883],
          [36849.24061946, 13757.68241714], [9028.01112208, 3628.67740715],
          [596.91685425, 110.08074646]]
    tt = [[705.48895748, 262.14055831], [6469.02581579, 2404.83678008],
          [36827.71983838, 13751.52554662], [9143.97447325, 3050.69467676],
          [662.31133378, 60.66609697]]
    tb = [[705.4903004, 262.13980169], [6469.19940132, 2404.75760398],
          [36849.2113973, 13757.8052238], [9010.83583125, 3630.48463588],
          [601.45212141, 111.52139435]]
    linear_power = np.array(p)
    truth_smith = np.array(ts)
    truth_takahashi = np.array(tt)
    truth_bird = np.array(tb)

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
    power_1d = linear_power[:, 0]
    truth_scalar_redshift = truth_smith[:, 0]
    smith_scalar_redshift = halofit_smith(k, z_scalar, power_1d, Planck15)
    assert allclose(smith_scalar_redshift, truth_scalar_redshift)
    assert np.shape(smith_scalar_redshift) == np.shape(power_1d)

    # Test for failure when wavenumber is a scalar
    k_scalar = k[0]
    power_1d = linear_power[0, :]
    with pytest.raises(TypeError):
        halofit_smith(k_scalar, z, power_1d, Planck15)

    # Test for failure when wavenumber array is the wrong size
    k_wrong_size = np.logspace(-4.0, 2.0, 7)
    with pytest.raises(ValueError):
        halofit_smith(k_wrong_size, z, linear_power, Planck15)

    # Test for failure when redshift arry is the wrong size
    z_wrong_size = np.linspace(0.0, 2.0, 3)
    with pytest.raises(TypeError):
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
