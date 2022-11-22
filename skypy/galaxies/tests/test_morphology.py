import numpy as np
import pytest
from scipy import stats

from astropy import units

from astropy.cosmology import FlatLambdaCDM


def test_angular_size():
    """ Test a FlatLambdaCDM cosmology with omega_matter = 1"""

    from skypy.galaxies import morphology

    cosmology = FlatLambdaCDM(Om0=1.0, H0=70.0)

    # Test that a scalar input gives a scalar output
    scalar_radius = 1.0 * units.kpc
    scalar_redshift = 1.0
    angular_size = morphology.angular_size(scalar_radius, scalar_redshift, cosmology)

    assert np.isscalar(angular_size.value)

    # Test that the output has the correct units
    assert angular_size.unit.is_equivalent(units.rad)

    # If the input have bad units, a UnitConversionError is raised
    radius_without_units = 1.0

    with pytest.raises(units.UnitTypeError):
        morphology.angular_size(radius_without_units, scalar_redshift, cosmology)


@pytest.mark.flaky
def test_beta_ellipticity():

    from skypy.galaxies.morphology import beta_ellipticity

    # randomised ellipticity distribution with beta distribution parameters a,b
    # and the equivalent reparametrisation
    a, b = np.random.lognormal(size=2)
    e_ratio, e_sum = (a / (a + b), a + b)

    # Test scalar output
    assert np.isscalar(beta_ellipticity(e_ratio, e_sum))

    # Test array output
    assert beta_ellipticity(e_ratio, e_sum, size=10).shape == (10,)

    # Test broadcast output
    e_ratio2 = 0.5 * np.ones((13, 1, 5))
    e_sum2 = 0.5 * np.ones((7, 5))
    rvs = beta_ellipticity(e_ratio2, e_sum2)
    assert rvs.shape == np.broadcast(e_ratio2, e_sum2).shape

    # Kolmogorov-Smirnov test comparing ellipticity and beta distributions
    D, p = stats.kstest(beta_ellipticity(e_ratio, e_sum, size=1000), 'beta',
                        args=(a, b))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # Kolmogorov-Smirnov test comparing ellipticity and uniform distributions
    D, p = stats.kstest(beta_ellipticity(0.5, 2.0, size=1000), 'uniform')
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # Kolmogorov-Smirnov test comparing ellipticity and arcsine distributions
    D, p = stats.kstest(beta_ellipticity(0.5, 1.0, size=1000), 'arcsine')
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)


@pytest.mark.flaky
def test_late_type_lognormal_size():
    """ Test lognormal distribution of late-type galaxy sizes"""

    from skypy.galaxies.morphology import late_type_lognormal_size

    # Test that a scalar input gives a scalar output
    magnitude_scalar = -20.0
    alpha, beta, gamma, M0 = 0.21, 0.53, -1.31, -20.52
    sigma1, sigma2 = 0.48, 0.25
    size_scalar = late_type_lognormal_size(magnitude_scalar, alpha, beta,
                                           gamma, M0, sigma1, sigma2)

    assert np.isscalar(size_scalar.value)

    # Test that the output has the correct units
    assert size_scalar.unit.is_equivalent(units.kpc)

    # Test that an array input gives an array output, with the same shape
    magnitude_array = np.array([-20.0, -21.0])
    size_array = late_type_lognormal_size(magnitude_array, alpha, beta,
                                          gamma, M0, sigma1, sigma2)

    assert np.shape(size_array.value) == np.shape(magnitude_array)

    # Test that size not None gives an array output, with the correct shape
    size_sample = late_type_lognormal_size(magnitude_scalar, alpha, beta,
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
def test_early_type_lognormal_size():
    """ Test lognormal distribution of late-type galaxy sizes"""

    from skypy.galaxies.morphology import early_type_lognormal_size

    # Test that a scalar input gives a scalar output
    magnitude_scalar = -20.0
    a, b, M0 = 0.6, -4.63, -20.52
    sigma1, sigma2 = 0.48, 0.25
    size_scalar = early_type_lognormal_size(magnitude_scalar, a, b, M0,
                                            sigma1, sigma2)

    assert np.isscalar(size_scalar.value)

    # Test that the output has the correct units
    assert size_scalar.unit.is_equivalent(units.kpc)

    # Test that an array input gives an array output, with the same shape
    magnitude_array = np.array([-20.0, -21.0])
    size_array = early_type_lognormal_size(magnitude_array, a, b, M0,
                                           sigma1, sigma2)

    assert np.shape(size_array.value) == np.shape(magnitude_array)

    # Test that size not None gives an array output, with the correct shape
    size_sample = early_type_lognormal_size(magnitude_scalar, a, b, M0,
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
def test_linear_lognormal_size():
    """ Test lognormal distribution of galaxy sizes"""

    from skypy.galaxies.morphology import linear_lognormal_size

    # Test that a scalar input gives a scalar output
    magnitude_scalar = -20.0
    a_mu, b_mu, sigma = -0.24, -4.63, 0.4
    size_scalar = linear_lognormal_size(magnitude_scalar, a_mu, b_mu, sigma)

    assert np.isscalar(size_scalar.value)

    # Test that the output has the correct units
    assert size_scalar.unit.is_equivalent(units.kpc)

    # Test that an array input gives an array output, with the same shape
    magnitude_array = np.array([-20.0, -21.0])
    size_array = linear_lognormal_size(magnitude_array, a_mu, b_mu, sigma)

    assert np.shape(size_array.value) == np.shape(magnitude_array)

    # Test that size not None gives an array output, with the correct shape
    size_sample = linear_lognormal_size(magnitude_scalar, a_mu, b_mu,
                                        sigma, size=1000)

    assert np.shape(size_sample.value) == (1000,)

    # Test the distribution of galaxy sizes follows a lognormal distribution
    mean = a_mu * magnitude_scalar + b_mu
    arguments = (sigma, 0, np.power(10, mean))
    d, p = stats.kstest(size_sample, 'lognorm', args=arguments)

    assert p > 0.01


def test_ryden04_ellipticity():
    from skypy.galaxies.morphology import ryden04_ellipticity

    # sample a single ellipticity
    e = ryden04_ellipticity(0.222, 0.056, -1.85, 0.89)
    assert np.isscalar(e)

    # sample many ellipticities
    e = ryden04_ellipticity(0.222, 0.056, -1.85, 0.89, size=1000)
    assert np.shape(e) == (1000,)

    # sample with explicit shape
    e = ryden04_ellipticity(0.222, 0.056, -1.85, 0.89, size=(10, 10))
    assert np.shape(e) == (10, 10)

    # sample with implicit size
    e1 = ryden04_ellipticity([0.222, 0.333], 0.056, -1.85, 0.89)
    e2 = ryden04_ellipticity(0.222, [0.056, 0.067], -1.85, 0.89)
    e3 = ryden04_ellipticity(0.222, 0.056, [-1.85, -2.85], 0.89)
    e4 = ryden04_ellipticity(0.222, 0.056, -1.85, [0.89, 1.001])
    assert np.shape(e1) == np.shape(e2) == np.shape(e3) == np.shape(e4) == (2,)

    # sample with broadcasting rule
    e = ryden04_ellipticity([[0.2, 0.3], [0.4, 0.5]], 0.1, [-1.9, -2.9], 0.9)
    assert np.shape(e) == (2, 2)

    # sample with random parameters and check that result is in unit range
    args = np.random.rand(4)*[1., .1, -2., 1.]
    e = ryden04_ellipticity(*args, size=1000)
    assert np.all((e >= 0.) & (e <= 1.))

    # sample a spherical distribution
    e = ryden04_ellipticity(1-1e-99, 1e-99, -1e99, 1e-99, size=1000)
    assert np.allclose(e, 0.)
