import numpy as np
from astropy import units
from scipy import stats
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
    """ Test lognormal distribution of galaxy sizes"""
    # Test that a scalar input gives a scalar output
    scalar_magnitude = 26.0
    a_mu = 1.0 * units.kpc
    b_mu = 0.0 * units.kpc
    sigma = 1.0 * units.kpc
    size_output = size.linear_lognormal(scalar_magnitude, a_mu, b_mu, sigma)

    assert np.isscalar(size_output.value)

    # Test the distribution of galaxy sizes follows a lognormal distribution
    size_distribution = size.linear_lognormal(scalar_magnitude, a_mu, b_mu,
                                              sigma, size=1000).value

    mu_physical = a_mu * scalar_magnitude + b_mu
    mu_value = mu_physical.value

    arguments = (sigma.value, 0, np.exp(mu_value))
    test = stats.kstest(size_distribution, 'lognorm', args=arguments)

    D_value = test[0]
    p_value = test[1]

    assert p_value > 0.01
    assert D_value < p_value

    # Test the returned distribution is of the right shape
    assert size_distribution.shape == (1000,)
