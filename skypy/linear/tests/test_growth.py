import numpy as np
from astropy.cosmology import FlatLambdaCDM, default_cosmology, Planck15
from astropy.units import isclose, allclose
import pytest

from skypy.linear import growth


def test_carroll():
    """ Test a FlatLambdaCDM cosmology with omega_matter = 1"""
    cosmology = FlatLambdaCDM(Om0=1.0, H0=70.0)

    # Test that a scalar input gives a scalar output
    scalar_input = 1
    scalar_output = growth.growth_function_carroll(scalar_input, cosmology)
    assert np.isscalar(scalar_output)

    # Test that an array input gives an array output
    array_shape = (10,)
    array_input = np.random.uniform(size=array_shape)
    array_output = growth.growth_function_carroll(array_input, cosmology)
    assert array_output.shape == array_shape

    # Test against theory for omega_matter = 1.0
    redshift = np.linspace(0.0, 10.0, 100)
    Dz_carroll = growth.growth_function_carroll(redshift, cosmology)
    Dz_theory = 1.0 / (1.0 + redshift)
    assert allclose(Dz_carroll, Dz_theory)

    # Test against precomputed values for Planck15
    redshift = np.linspace(0, 5, 4)
    Dz_carroll = growth.growth_function_carroll(redshift, Planck15)
    Dz_truth = np.array([0.78136173, 0.36635322, 0.22889793, 0.16577711])
    assert allclose(Dz_carroll, Dz_truth)

    # Test for failure when redshift < 0
    negative_redshift_scalar = -1
    with pytest.raises(ValueError):
        growth.growth_function_carroll(negative_redshift_scalar, cosmology)
    negative_redshift_array = [0, 1, -2, 3]
    with pytest.raises(ValueError):
        growth.growth_function_carroll(negative_redshift_array, cosmology)


def test_growth():
    """
    Test a FlatLambdaCDM cosmology with omega_matter = 1.0
    and astropy default cosmology
    """

    redshift = np.linspace(0., 10., 101)
    cosmology_flat = FlatLambdaCDM(H0=70.0, Om0=1.0)

    fz = growth.growth_factor(redshift, cosmology_flat)
    Dz = growth.growth_function(redshift, cosmology_flat)
    Dzprime = growth.growth_function_derivative(redshift, cosmology_flat)

    # Test growth factor
    assert redshift.shape == fz.shape,\
        "Length of redshift array and growth rate array do not match"
    assert isclose(fz[0], 1.0),\
        "Growth factor at redshift 0 is not close to 1.0"

    # Test growth function
    assert redshift.shape == Dz.shape,\
        "Length of redshift array and growth function array do not match"
    assert allclose(Dz, 1. / (1. + redshift)),\
        "Growth function is not close to the scale factor"

    # Test growth function derivative
    assert redshift.shape == Dzprime.shape,\
        "Length of redshift array and growth function array do not match"
    assert isclose(Dzprime[0], -1.0),\
        "Derivative of growth function at redshift 0 is not close to -1.0"

    # Test against precomputed values using astropy default cosmology
    default = default_cosmology.get()
    zvec = np.linspace(0.0, 1.0, 4)

    fz_default = growth.growth_factor(zvec, default)
    Dz_default = growth.growth_function(zvec, default)
    Dzprime_default = growth.growth_function_derivative(zvec, default)

    precomputed_fz_default = np.array([0.5255848, 0.69412802, 0.80439553,
                                       0.87179376])
    precomputed_Dz_default = np.array([0.66328939, 0.55638978, 0.4704842,
                                       0.40368459])
    precomputed_Dzprime_default = np.array([-0.34861482, -0.2896543,
                                            -0.22707323, -0.17596485])

    assert allclose(fz_default, precomputed_fz_default)
    assert allclose(Dz_default, precomputed_Dz_default)
    assert allclose(Dzprime_default, precomputed_Dzprime_default)
