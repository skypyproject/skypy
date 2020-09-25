import numpy as np
import pytest
from scipy.stats import kstest


def test_magnitude_functions():

    from astropy.cosmology import default_cosmology

    from skypy.galaxy.luminosity import (absolute_to_apparent_magnitude,
            apparent_to_absolute_magnitude, distance_modulus,
            luminosity_in_band, luminosity_from_absolute_magnitude,
            absolute_magnitude_from_luminosity)

    cosmo = default_cosmology.get()

    # sample some redshifts
    z = np.random.uniform(0, 10, size=1000)

    # sample some absolute magnitudes
    M = np.random.uniform(15, 25, size=1000)

    # sample distance moduli
    DM = cosmo.distmod(z).value

    # compare with function
    np.testing.assert_allclose(distance_modulus(z), DM)

    # compute apparent magnitudes
    m = absolute_to_apparent_magnitude(M, DM)

    # compare with values
    np.testing.assert_allclose(m, M+DM)

    # go back to absolute magnitudes
    M_ = apparent_to_absolute_magnitude(m, DM)

    # compare with original values
    np.testing.assert_allclose(M_, M)

    # convert between absolute luminosity and magnitude
    assert np.isclose(luminosity_from_absolute_magnitude(-22), 630957344.5)
    assert np.isclose(absolute_magnitude_from_luminosity(630957344.5), -22)

    # convert with standard luminosities
    for ref, mag in luminosity_in_band.items():
        assert np.isclose(luminosity_from_absolute_magnitude(mag, ref), 1.0)
        assert np.isclose(absolute_magnitude_from_luminosity(1.0, ref), mag)

    # error when unknown reference is used
    with pytest.raises(KeyError):
        luminosity_from_absolute_magnitude(0., 'unknown')
    with pytest.raises(KeyError):
        absolute_magnitude_from_luminosity(1., 'unknown')


@pytest.mark.flaky
def test_schechter_lf_magnitude():
    from skypy.galaxy.luminosity import schechter_lf_magnitude
    from astropy.cosmology import default_cosmology
    import pytest

    # use default cosmology
    cosmo = default_cosmology.get()

    # Schechter function parameters for tests
    M_star = -20.5
    alpha = -1.3

    # sample 1000 galaxies at a fixed redshift of 1.0
    z = np.repeat(1.0, 1000)
    M = schechter_lf_magnitude(z, M_star, alpha, 30., cosmo)

    # get the distribution function
    log10_x_min = -0.4*(30. - cosmo.distmod(1.0).value - M_star)
    x = np.logspace(log10_x_min, log10_x_min + 3, 1000)
    pdf = x**(alpha+1)*np.exp(-x)
    cdf = np.concatenate([[0.], np.cumsum((pdf[1:]+pdf[:-1])/2*np.diff(np.log(x)))])
    cdf /= cdf[-1]

    # test the samples against the CDF
    D, p = kstest(10.**(-0.4*(M - M_star)), lambda t: np.interp(t, x, cdf))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # test for 1000 galaxies with Pareto redshift distribution
    z = np.random.pareto(3., size=1000)

    # for scalar parameters, sample galaxies with magnitude limit of 30
    M = schechter_lf_magnitude(z, M_star, alpha, 30., cosmo)

    # check that the output has the correct shape
    assert np.shape(M) == (1000,)

    # make sure magnitude limit was respected
    M_lim = 30. - cosmo.distmod(z).value
    assert np.all(M <= M_lim)

    # sample with array for alpha
    # not implemented at the moment
    with pytest.raises(NotImplementedError):
        M = schechter_lf_magnitude(z, M_star, np.broadcast_to(alpha, z.shape), 30., cosmo)

    # sample with an explicit size
    schechter_lf_magnitude(1.0, M_star, alpha, 30., cosmo, size=100)
