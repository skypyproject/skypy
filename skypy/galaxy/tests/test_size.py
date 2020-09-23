import numpy as np
from astropy import units
from scipy import stats
import pytest

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

    # Test that the output has the correct units
    assert angular_size.unit.is_equivalent(units.rad)

    # If the input have bad units, a UnitConversionError is raised
    radius_without_units = 1.0

    with pytest.raises(units.UnitTypeError):
        size.angular_size(radius_without_units, scalar_redshift, cosmology)


@pytest.mark.flaky
def test_late_type_lognormal():
    """ Test lognormal distribution of late-type galaxy sizes"""
    # Test that a scalar input gives a scalar output
    magnitude_scalar = -20.0
    alpha, beta, gamma, M0 = 0.21, 0.53, -1.31, -20.52
    sigma1, sigma2 = 0.48, 0.25
    size_scalar = size.late_type_lognormal(magnitude_scalar, alpha, beta,
                                           gamma, M0, sigma1, sigma2)

    assert np.isscalar(size_scalar.value)

    # Test that the output has the correct units
    assert size_scalar.unit.is_equivalent(units.kpc)

    # Test that an array input gives an array output, with the same shape
    magnitude_array = np.array([-20.0, -21.0])
    size_array = size.late_type_lognormal(magnitude_array, alpha, beta, gamma,
                                          M0, sigma1, sigma2)

    assert np.shape(size_array.value) == np.shape(magnitude_array)

    # Test that size not None gives an array output, with the correct shape
    size_sample = size.late_type_lognormal(magnitude_scalar, alpha, beta,
                                           gamma, M0, sigma1, sigma2,
                                           size=1000)

    assert np.shape(size_sample.value) == (1000,)

    # Test the distribution of galaxy sizes follows a lognormal distribution
    mean = -0.4 * alpha * magnitude_scalar + (beta - alpha) *\
        np.log10(1 + np.power(10, -0.4 * (magnitude_scalar - M0)))\
        + gamma
    sigma = sigma2 + (sigma1 - sigma2) /\
        (1.0 + np.power(10, -0.8 * (magnitude_scalar - M0)))

    arguments = (sigma, 0, np.power(10, mean))
    d, p = stats.kstest(size_sample, 'lognorm', args=arguments)

    assert p > 0.01


@pytest.mark.flaky
def test_early_type_lognormal():
    """ Test lognormal distribution of late-type galaxy sizes"""
    # Test that a scalar input gives a scalar output
    magnitude_scalar = -20.0
    a, b, M0 = 0.6, -4.63, -20.52
    sigma1, sigma2 = 0.48, 0.25
    size_scalar = size.early_type_lognormal(magnitude_scalar, a, b, M0,
                                            sigma1, sigma2)

    assert np.isscalar(size_scalar.value)

    # Test that the output has the correct units
    assert size_scalar.unit.is_equivalent(units.kpc)

    # Test that an array input gives an array output, with the same shape
    magnitude_array = np.array([-20.0, -21.0])
    size_array = size.early_type_lognormal(magnitude_array, a, b, M0,
                                           sigma1, sigma2)

    assert np.shape(size_array.value) == np.shape(magnitude_array)

    # Test that size not None gives an array output, with the correct shape
    size_sample = size.early_type_lognormal(magnitude_scalar, a, b, M0,
                                            sigma1, sigma2, size=1000)

    assert np.shape(size_sample.value) == (1000,)

    # Test the distribution of galaxy sizes follows a lognormal distribution
    mean = -0.4 * a * magnitude_scalar + b
    sigma = sigma2 + (sigma1 - sigma2) /\
                     (1.0 + np.power(10, -0.8 * (magnitude_scalar - M0)))

    arguments = (sigma, 0, np.power(10, mean))
    d, p = stats.kstest(size_sample, 'lognorm', args=arguments)

    assert p > 0.01


@pytest.mark.flaky
def test_linear_lognormal():
    """ Test lognormal distribution of galaxy sizes"""
    # Test that a scalar input gives a scalar output
    magnitude_scalar = -20.0
    a_mu, b_mu, sigma = -0.24, -4.63, 0.4
    size_scalar = size.linear_lognormal(magnitude_scalar, a_mu, b_mu, sigma)

    assert np.isscalar(size_scalar.value)

    # Test that the output has the correct units
    assert size_scalar.unit.is_equivalent(units.kpc)

    # Test that an array input gives an array output, with the same shape
    magnitude_array = np.array([-20.0, -21.0])
    size_array = size.linear_lognormal(magnitude_array, a_mu, b_mu, sigma)

    assert np.shape(size_array.value) == np.shape(magnitude_array)

    # Test that size not None gives an array output, with the correct shape
    size_sample = size.linear_lognormal(magnitude_scalar, a_mu, b_mu, sigma,
                                        size=1000)

    assert np.shape(size_sample.value) == (1000,)

    # Test the distribution of galaxy sizes follows a lognormal distribution
    mean = a_mu * magnitude_scalar + b_mu
    arguments = (sigma, 0, np.power(10, mean))
    d, p = stats.kstest(size_sample, 'lognorm', args=arguments)

    assert p > 0.01
