from astropy.cosmology import FlatLambdaCDM, Planck15
from astropy.units import allclose
import numpy as np
import pytest
from skypy.linear.growth import growth_function_carroll


def test_carroll():
    """ Test a FlatLambdaCDM cosmology with omega_matter = 1"""
    cosmology = FlatLambdaCDM(Om0=1.0, H0=70.0)

    # Test that a scalar input gives a scalar output
    scalar_input = 1
    scalar_output = growth_function_carroll(scalar_input, cosmology)
    assert np.isscalar(scalar_output)

    # Test that an array input gives an array output
    array_shape = (10,)
    array_input = np.random.uniform(size=array_shape)
    array_output = growth_function_carroll(array_input, cosmology)
    assert array_output.shape == array_shape

    # Test against theory for omega_matter = 1.0
    redshift = np.linspace(0.0, 10.0, 100)
    Dz_carroll = growth_function_carroll(redshift, cosmology)
    Dz_theory = 1.0 / (1.0 + redshift)
    assert allclose(Dz_carroll, Dz_theory)

    # Test against precomputed values for Planck15
    redshift = np.linspace(0,5,4)
    Dz_carroll = growth_function_carroll(redshift, Planck15)
    Dz_truth = np.array([0.78136173, 0.36635322, 0.22889793, 0.16577711])
    assert allclose(Dz_carroll, Dz_truth)

    # Test for failure when redshift < 0
    negative_redshift_scalar = -1
    with pytest.raises(ValueError):
        growth_function_carroll(negative_redshift_scalar, cosmology)
    negative_redshift_array = [0, 1, -2, 3]
    with pytest.raises(ValueError):
        growth_function_carroll(negative_redshift_array, cosmology)
