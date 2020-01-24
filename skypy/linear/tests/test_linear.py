from astropy.cosmology import FlatLambdaCDM
from astropy.units import allclose
import numpy as np
from skypy.linear.growth import carroll


def test_carroll():
    """ Test a FlatLambdaCDM cosmology with omega_matter = 1"""
    redshift = np.linspace(0., 10., 101)
    cosmology = FlatLambdaCDM(Om0=1., H0=70.)
    growth = carroll(redshift, cosmology)
    assert redshift.shape == growth.shape
    assert allclose(growth, 1./(1.+redshift))
