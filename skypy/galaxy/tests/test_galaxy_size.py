import numpy as np
from astropy.cosmology import FlatLambdaCDM

from skypy.galaxy import galaxy_size as gs


def test_galaxy_size():
    """ Test a FlatLambdaCDM cosmology with omega_matter = 1"""
    cosmology = FlatLambdaCDM(Om0=1.0, H0=70.0)

    # Test that a scalar input gives a scalar output
    scalar_radius = 1.0
    scalar_redshift = 1.0
    angular_size = gs.angular_size(scalar_radius, scalar_redshift, cosmology)

    scalar_magnitude = 1.0
    half_light_angular_size = gs.half_light_angular_size(scalar_radius,
                                                         scalar_magnitude,
                                                         cosmology)

    assert np.isscalar(angular_size)
    assert np.isscalar(half_light_angular_size)

    # Test that an array input gives an array output
    array_radius = np.logspace(-3, 1, num=1000)
    array_redshift = np.logspace(-2, 0, num=1000)
    array_angular_size = gs.angular_size(array_radius, array_redshift,
                                         cosmology)

    array_magnitude = np.linspace(-26, 26, num=1000)
    array_half_light_angular_size = gs.half_light_angular_size(array_radius,
                                                               array_magnitude,
                                                               cosmology)

    nr = array_radius
    nz = array_redshift
    nM = array_magnitude

    assert array_angular_size.shape == (nz, nr)
    assert array_half_light_angular_size.shape == (nz, nM)
