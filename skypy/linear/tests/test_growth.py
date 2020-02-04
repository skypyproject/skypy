import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.units import isclose, allclose

from skypy.linear import growth


def test_growth():
    """ Test a FlatLambdaCDM cosmology with omega_matter = 1.0 """

    redshift = np.linspace(0., 10., 101)
    cosmology = FlatLambdaCDM(H0=70.0, Om0=1.0)

    fz = growth.growth_factor(redshift, cosmology)
    Dz = growth.growth_function(redshift, cosmology)
    Dzprime = growth.growth_function_derivative(redshift, cosmology)

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
