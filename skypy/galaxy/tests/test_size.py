import numpy as np
from astropy import units
from astropy.cosmology import FlatLambdaCDM

from skypy.galaxy import size


def test_angular_size():
    """ Test a FlatLambdaCDM cosmology with omega_matter = 1"""
    cosmology = FlatLambdaCDM(Om0=1.0, H0=70.0)

    # Test that a scalar input gives a scalar output
    scalar_radius = 1.0 * units.kpc
    scalar_redshift = 1.0
    angular_size = size.angular_size(scalar_radius, scalar_redshift, cosmology)

    assert np.isscalar(angular_size.value)


def test_linear_lognormal():
    """ Test lognormal distribution of galaxy radii"""
    # Test that a scalar input gives a scalar output
    scalar_magnitude = 1.0
    lognormal = size.linear_lognormal(scalar_magnitude)

    assert np.isscalar(lognormal.value)
