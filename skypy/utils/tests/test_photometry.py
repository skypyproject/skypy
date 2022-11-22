import numpy as np
import pytest
from skypy.utils.photometry import HAS_SPECLITE


def test_magnitude_functions():

    from skypy.utils.photometry import (luminosity_in_band,
                                        luminosity_from_absolute_magnitude,
                                        absolute_magnitude_from_luminosity)

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


@pytest.mark.skipif(not HAS_SPECLITE, reason='test requires speclite')
def test_mag_ab_standard_source():

    from astropy import units
    from speclite.filters import FilterResponse
    from skypy.utils.photometry import mag_ab

    # create a filter
    filt_lam = np.logspace(0, 4, 1000)*units.AA
    filt_tx = np.exp(-((filt_lam - 1000*units.AA)/(100*units.AA))**2)
    filt_tx[[0, -1]] = 0
    FilterResponse(wavelength=filt_lam, response=filt_tx,
                   meta=dict(group_name='test', band_name='filt'))

    # test that the AB standard source has zero magnitude
    lam = filt_lam  # same grid to prevent interpolation issues
    flam = 0.10885464149979998*units.Unit('erg s-1 cm-2 AA')/lam**2

    m = mag_ab(lam, flam, 'test-filt')

    assert np.isclose(m, 0)


@pytest.mark.skipif(not HAS_SPECLITE, reason='test requires speclite')
def test_mag_ab_redshift_dependence():

    from astropy import units
    from speclite.filters import FilterResponse
    from skypy.utils.photometry import mag_ab

    # make a wide tophat bandpass
    filt_lam = [1.0e-10, 1.1e-10, 1.0e0, 0.9e10, 1.0e10]
    filt_tx = [0., 1., 1., 1., 0.]
    FilterResponse(wavelength=filt_lam, response=filt_tx,
                   meta=dict(group_name='test', band_name='filt'))

    # create a narrow gaussian source
    lam = np.logspace(-11, 11, 1000)*units.AA
    flam = np.exp(-((lam - 100*units.AA)/(10*units.AA))**2)*units.Unit('erg s-1 cm-2 AA-1')

    # array of redshifts
    z = np.linspace(0, 1, 11)

    # compute the AB magnitude at different redshifts
    m = mag_ab(lam, flam, 'test-filt', redshift=z)

    # compare with expected redshift dependence
    np.testing.assert_allclose(m, m[0] - 2.5*np.log10(1 + z))


@pytest.mark.skipif(not HAS_SPECLITE, reason='test requires speclite')
def test_mag_ab_multi():

    from astropy import units
    from skypy.utils.photometry import mag_ab
    from speclite.filters import FilterResponse

    # 5 redshifts
    z = np.linspace(0, 1, 5)

    # 2 Gaussian bandpasses
    filt_lam = np.logspace(0, 4, 1000) * units.AA
    filt_mean = np.array([[1000], [2000]]) * units.AA
    filt_width = np.array([[100], [10]]) * units.AA
    filt_tx = np.exp(-((filt_lam-filt_mean)/filt_width)**2)
    filt_tx[:, [0, -1]] = 0
    FilterResponse(wavelength=filt_lam, response=filt_tx[0],
                   meta=dict(group_name='test', band_name='filt0'))
    FilterResponse(wavelength=filt_lam, response=filt_tx[1],
                   meta=dict(group_name='test', band_name='filt1'))

    # 3 Flat Spectra
    # to prevent issues with interpolation, collect all redshifted filt_lam
    lam = []
    for z_ in z:
        lam = np.union1d(lam, filt_lam.value/(1+z_))
    lam = lam*filt_lam.unit
    A = np.array([[2], [3], [4]])
    flam = A * 0.10885464149979998*units.Unit('erg s-1 cm-2 AA')/lam**2

    # Compare calculated magnitudes with truth
    magnitudes = mag_ab(lam, flam, ['test-filt0', 'test-filt1'], redshift=z)
    truth = -2.5 * np.log10(A * (1+z)).T[:, :, np.newaxis]
    assert magnitudes.shape == (5, 3, 2)
    np.testing.assert_allclose(*np.broadcast_arrays(magnitudes, truth), rtol=1e-4)


@pytest.mark.skipif(not HAS_SPECLITE, reason='test requires speclite')
def test_template_spectra():

    from astropy import units
    from skypy.utils.photometry import mag_ab, SpectrumTemplates
    from astropy.cosmology import Planck15
    from speclite.filters import FilterResponse

    class TestTemplates(SpectrumTemplates):
        '''Three flat templates'''

        def __init__(self):
            self.wavelength = np.logspace(-1, 4, 1000)*units.AA
            A = np.array([[2], [3], [4]]) * 0.10885464149979998
            self.templates = A * units.Unit('erg s-1 cm-2 AA') / self.wavelength**2

    test_templates = TestTemplates()
    lam, flam = test_templates.wavelength, test_templates.templates

    # Gaussian bandpass
    filt_lam = np.logspace(0, 4, 1000)*units.AA
    filt_tx = np.exp(-((filt_lam - 1000*units.AA)/(100*units.AA))**2)
    filt_tx[[0, -1]] = 0
    FilterResponse(wavelength=filt_lam, response=filt_tx,
                   meta=dict(group_name='test', band_name='filt'))

    # Each test galaxy is exactly one of the templates
    coefficients = np.eye(3)

    # Test absolute magnitudes
    mt = test_templates.absolute_magnitudes(coefficients, 'test-filt')
    m = mag_ab(lam, flam, 'test-filt')
    np.testing.assert_allclose(mt, m)

    # Test apparent magnitudes
    redshift = np.array([1, 2, 3])
    dm = Planck15.distmod(redshift).value
    mt = test_templates.apparent_magnitudes(coefficients, redshift, 'test-filt', Planck15)
    np.testing.assert_allclose(mt, m - 2.5*np.log10(1+redshift) + dm)

    # Redshift interpolation test; linear interpolation sufficient over a small
    # redshift range at low relative tolerance
    z = np.linspace(0.1, 0.2, 3)
    m_true = test_templates.apparent_magnitudes(coefficients, z, 'test-filt',
                                                Planck15, resolution=4)
    m_interp = test_templates.apparent_magnitudes(coefficients, z, 'test-filt',
                                                  Planck15, resolution=2)
    np.testing.assert_allclose(m_true, m_interp, rtol=1e-5)
    assert not np.all(m_true == m_interp)


@pytest.mark.skipif(HAS_SPECLITE, reason='test requires that speclite is not installed')
def test_speclite_not_installed():
    """
    Regression test for #436
    Test that mag_ab raises the correct exception if speclite is not insalled.
    """
    from skypy.utils.photometry import mag_ab
    wavelength = np.linspace(1, 10, 100)
    spectrum = np.ones(10)
    filter = 'bessell-B'
    with pytest.raises(ImportError):
        mag_ab(wavelength, spectrum, filter)


def test_magnitude_error_rykoff():
    from skypy.utils.photometry import magnitude_error_rykoff

    # Test broadcasting to same shape given array for each parameter and
    # test for correct result.
    magnitude = np.full((2, 1, 1, 1, 1), 21)
    magnitude_limit = np.full((3, 1, 1, 1), 21)
    magnitude_zp = np.full((5, 1, 1), 21)
    a = np.full((7, 1), np.log(200))
    b = np.zeros(11)
    error = magnitude_error_rykoff(magnitude, magnitude_limit, magnitude_zp, a, b)
    # test result
    assert np.allclose(error, 0.25 / np.log(10))
    # test shape
    assert error.shape == (2, 3, 5, 7, 11)

    # second test for result
    magnitude = 20
    magnitude_limit = 22.5
    magnitude_zp = 25
    b = 2
    a = np.log(10) - 1.5 * b
    error = magnitude_error_rykoff(magnitude, magnitude_limit, magnitude_zp, a, b)
    assert np.isclose(error, 0.25 / np.log(10) / np.sqrt(10))

    # test that error limit is returned if error is larger than error_limit
    # The following set-up would give a value larger than 10
    magnitude = 30
    magnitude_limit = 25
    magnitude_zp = 30
    a = 0.5
    b = 1.0
    error_limit = 1
    error = magnitude_error_rykoff(magnitude, magnitude_limit, magnitude_zp, a, b, error_limit)
    assert error == error_limit


def test_logistic_completeness_function():
    from skypy.utils.photometry import logistic_completeness_function

    # Test that arguments broadcast correctly
    m = np.full((2, 1, 1), 21)
    m95 = np.full((3, 1), 22)
    m50 = np.full(5, 23)
    p = logistic_completeness_function(m, m95, m50)
    assert p.shape == np.broadcast(m, m95, m50).shape

    # Test result of completeness function for different given magnitudes
    m95 = 24
    m50 = 25
    m = [np.finfo(np.float64).min, m95, m50, 2*m50-m95, np.finfo(np.float64).max]
    p = logistic_completeness_function(m, m95, m50)
    assert np.allclose(p, [1, 0.95, 0.5, 0.05, 0])
