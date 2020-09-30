import numpy as np
import scipy.stats
import pytest

try:
    import specutils
except ImportError:
    HAS_SPECUTILS = False
else:
    HAS_SPECUTILS = True


@pytest.mark.flaky
def test_sampling_coefficients():

    from skypy.galaxy.spectrum import dirichlet_coefficients

    alpha0 = np.array([2.079, 3.524, 1.917, 1.992, 2.536])
    alpha1 = np.array([2.265, 3.862, 1.921, 1.685, 2.480])

    z1 = 1.
    redshift = np.full(1000, 2.0, dtype=float)
    redshift_reshape = np.atleast_1d(redshift)[:, np.newaxis]
    alpha = np.power(alpha0, 1. - redshift_reshape / z1) * \
        np.power(alpha1, redshift_reshape / z1)

    a0 = alpha.sum(axis=1)

    # Check the output shape if redshift is an array
    coefficients = dirichlet_coefficients(redshift, alpha0, alpha1, z1)
    assert coefficients.shape == (len(redshift), len(alpha0)), \
        'Shape of coefficients array is not (len(redshift), len(alpha0)) '

    # the marginalised distributions are beta distributions with a = alpha_i
    # and b = a0-alpha_i
    for a, c in zip(alpha.T, coefficients.T):
        d, p = scipy.stats.kstest(c, 'beta', args=(a, a0 - a))
        assert p >= 0.01, \
            'Not all marginal distributions follow a beta distribution.'

    # test sampling with weights
    weight = [3.47e+09, 3.31e+06, 2.13e+09, 1.64e+10, 1.01e+09]
    coefficients = dirichlet_coefficients(redshift, alpha0, alpha1, z1, weight)
    assert coefficients.shape == (len(redshift), len(alpha0)), \
        'Shape of coefficients array is not (len(redshift), len(alpha0)) '

    # Test output shape if redshift is a scalar
    redshift = 2.0
    coefficients = dirichlet_coefficients(redshift, alpha0, alpha1)
    assert coefficients.shape == (len(alpha0),), \
        'Shape of coefficients array is not (len(alpha0),) ' \
        'if redshift array is float.'

    # Test raising ValueError of alpha1 and alpha0 have different size
    alpha0 = np.array([1, 2, 3])
    alpha1 = np.array([4, 5])
    redshift = np.linspace(0, 2, 10)
    with pytest.raises(ValueError):
        dirichlet_coefficients(redshift, alpha0, alpha1)

    # Test that ValueError is risen if alpha0 or alpha1 is a scalar.
    scalar_alpha = 1.
    with pytest.raises(ValueError):
        dirichlet_coefficients(redshift, scalar_alpha, alpha1)
    with pytest.raises(ValueError):
        dirichlet_coefficients(redshift, alpha0, scalar_alpha)

    # bad weight parameter
    with pytest.raises(ValueError):
        dirichlet_coefficients(redshift, [2.5, 2.5], [2.5, 2.5], weight=[1, 2, 3])


@pytest.mark.skipif(not HAS_SPECUTILS, reason='test requires specutils')
def test_mag_ab_standard_source():

    from astropy import units

    from skypy.galaxy.spectrum import mag_ab

    # create a bandpass
    bp_lam = np.logspace(0, 4, 1000)*units.AA
    bp_tx = np.exp(-((bp_lam - 1000*units.AA)/(100*units.AA))**2)*units.dimensionless_unscaled
    bp = specutils.Spectrum1D(spectral_axis=bp_lam, flux=bp_tx)

    # test that the AB standard source has zero magnitude
    lam = np.logspace(0, 4, 1000)*units.AA
    flam = 0.10884806248538730623*units.Unit('erg s-1 cm-2 AA')/lam**2
    spec = specutils.Spectrum1D(spectral_axis=lam, flux=flam)

    m = mag_ab(spec, bp)

    assert np.isclose(m, 0)


@pytest.mark.skipif(not HAS_SPECUTILS, reason='test requires specutils')
def test_mag_ab_redshift_dependence():

    from astropy import units

    from skypy.galaxy.spectrum import mag_ab

    # make a wide tophat bandpass
    bp_lam = np.logspace(-10, 10, 3)*units.AA
    bp_tx = np.ones(3)*units.dimensionless_unscaled
    bp = specutils.Spectrum1D(spectral_axis=bp_lam, flux=bp_tx)

    # create a narrow gaussian source
    lam = np.logspace(0, 3, 1000)*units.AA
    flam = np.exp(-((lam - 100*units.AA)/(10*units.AA))**2)*units.Unit('erg s-1 cm-2 AA-1')
    spec = specutils.Spectrum1D(spectral_axis=lam, flux=flam)

    # array of redshifts
    z = np.linspace(0, 1, 11)

    # compute the AB magnitude at different redshifts
    m = mag_ab(spec, bp, z)

    # compare with expected redshift dependence
    np.testing.assert_allclose(m, m[0] - 2.5*np.log10(1 + z))


@pytest.mark.skipif(not HAS_SPECUTILS, reason='test requires specutils')
def test_mag_ab_multi():

    from astropy import units
    from skypy.galaxy.spectrum import mag_ab

    # 5 redshifts
    z = np.linspace(0, 1, 5)

    # 2 Gaussian bandpasses
    bp_lam = np.logspace(0, 4, 1000) * units.AA
    bp_mean = np.array([[1000], [2000]]) * units.AA
    bp_width = np.array([[100], [10]]) * units.AA
    bp_tx = np.exp(-((bp_lam-bp_mean)/bp_width)**2)*units.dimensionless_unscaled
    bp = specutils.Spectrum1D(spectral_axis=bp_lam, flux=bp_tx)

    # 3 Flat Spectra
    lam = np.logspace(0, 4, 1000)*units.AA
    A = np.array([[2], [3], [4]])
    flam = A * 0.10884806248538730623*units.Unit('erg s-1 cm-2 AA')/lam**2
    spec = specutils.Spectrum1D(spectral_axis=lam, flux=flam)

    # Compare calculated magnitudes with truth
    magnitudes = mag_ab(spec, bp, z)
    truth = -2.5 * np.log10(A * (1+z)).T[:, np.newaxis, :]
    assert magnitudes.shape == (5, 2, 3)
    np.testing.assert_allclose(*np.broadcast_arrays(magnitudes, truth))


@pytest.mark.skipif(not HAS_SPECUTILS, reason='test requires specutils')
def test_template_spectra():

    from astropy import units
    from skypy.galaxy.spectrum import mag_ab, magnitudes_from_templates
    from astropy.cosmology import Planck15

    # 3 Flat Templates
    lam = np.logspace(0, 4, 1000)*units.AA
    A = np.array([[2], [3], [4]])
    flam = A * 0.10884806248538730623*units.Unit('erg s-1 cm-2 AA')/lam**2
    spec = specutils.Spectrum1D(spectral_axis=lam, flux=flam)

    # Gaussian bandpass
    bp_lam = np.logspace(0, 4, 1000)*units.AA
    bp_tx = np.exp(-((bp_lam - 1000*units.AA)/(100*units.AA))**2)*units.dimensionless_unscaled
    bp = specutils.Spectrum1D(spectral_axis=bp_lam, flux=bp_tx)

    # Each test galaxy is exactly one of the templates
    coefficients = np.diag(np.ones(3))
    mt = magnitudes_from_templates(coefficients, spec, bp)
    m = mag_ab(spec, bp)
    np.testing.assert_allclose(mt, m)

    # Test distance modulus
    redshift = np.array([0, 1, 2])
    dm = Planck15.distmod(redshift).value
    mt = magnitudes_from_templates(coefficients, spec, bp, distance_modulus=dm)
    np.testing.assert_allclose(mt, m + dm)

    # Test stellar mass
    sm = np.array([1, 2, 3])
    mt = magnitudes_from_templates(coefficients, spec, bp, stellar_mass=sm)
    np.testing.assert_allclose(mt, m - 2.5*np.log10(sm))

    # Redshift interpolation test; linear interpolation sufficient over a small
    # redshift range at low relative tolerance
    z = np.linspace(0, 0.1, 3)
    m_true = magnitudes_from_templates(coefficients, spec, bp, redshift=z, resolution=4)
    m_interp = magnitudes_from_templates(coefficients, spec, bp, redshift=z, resolution=2)
    np.testing.assert_allclose(m_true, m_interp, rtol=1e-2)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(m_true, m_interp, rtol=1e-5)


@pytest.mark.skipif(not HAS_SPECUTILS, reason='test requires specutils')
def test_stellar_mass_from_reference_band():

    from astropy import units
    from skypy.galaxy.spectrum import mag_ab, stellar_mass_from_reference_band

    # Gaussian bandpass
    bp_lam = np.logspace(0, 4, 1000) * units.AA
    bp_mean = 1000 * units.AA
    bp_width = 100 * units.AA
    bp_tx = np.exp(-((bp_lam-bp_mean)/bp_width)**2)*units.dimensionless_unscaled
    band = specutils.Spectrum1D(spectral_axis=bp_lam, flux=bp_tx)

    # 3 Flat template spectra
    lam = np.logspace(0, 4, 1000)*units.AA
    A = np.array([[2], [3], [4]])
    flam = A * 0.10884806248538730623*units.Unit('erg s-1 cm-2 AA')/lam**2
    templates = specutils.Spectrum1D(spectral_axis=lam, flux=flam)

    # Absolute magnitudes for each template
    Mt = mag_ab(templates, band)

    # Using the identity matrix for the coefficients yields trivial test cases
    coeff = np.diag(np.ones(3))

    # Using the absolute magnitudes of the templates as reference magnitudes
    # should return one solar mass for each template.
    stellar_mass = stellar_mass_from_reference_band(coeff, templates, Mt, band)
    truth = 1
    np.testing.assert_allclose(stellar_mass, truth)

    # Solution for given magnitudes without template mixing
    Mb = np.array([10, 20, 30])
    stellar_mass = stellar_mass_from_reference_band(coeff, templates, Mb, band)
    truth = np.power(10, -0.4*(Mb-Mt))
    np.testing.assert_allclose(stellar_mass, truth)


@pytest.mark.skipif(not HAS_SPECUTILS, reason='test requires specutils')
def test_load_spectral_data():

    from skypy.galaxy.spectrum import load_spectral_data
    from astropy.utils.data import get_pkg_data_filename

    # load a local file
    filename = get_pkg_data_filename('data/spectrum.ecsv')
    load_spectral_data(filename)

    # load skypy data bandpasses
    load_spectral_data('Johnson_UBV')
    load_spectral_data('Cousins_RI')

    # load skypy data spectrum templates
    load_spectral_data('kcorrect_spec')

    # load DECam bandpasses
    load_spectral_data('DECam_grizY')

    # load multiple sources
    load_spectral_data(['Johnson_B', 'Cousins_R'])

    # try to load non-string name
    with pytest.raises(TypeError):
        load_spectral_data(1.0)

    # try to load unknown data
    with pytest.raises(FileNotFoundError):
        load_spectral_data('!!UNKNOWN!!')


@pytest.mark.skipif(not HAS_SPECUTILS, reason='test requires specutils')
def test_combine_spectra():

    from skypy.galaxy._spectrum_loaders import combine_spectra
    from astropy import units

    a = specutils.Spectrum1D(spectral_axis=[1., 2., 3.]*units.AA,
                             flux=[1., 2., 3.]*units.Jy)
    b = specutils.Spectrum1D(spectral_axis=[1e-10, 2e-10, 3e-10]*units.m,
                             flux=[4e-23, 5e-23, 6e-23]*units.Unit('erg s-1 cm-2 Hz-1'))

    assert np.allclose(a.spectral_axis, b.spectral_axis, atol=0, rtol=1e-10)

    assert a == combine_spectra(a, None)
    assert a == combine_spectra(None, a)

    ab = combine_spectra(a, b)
    assert isinstance(ab, specutils.Spectrum1D)
    assert ab.shape == (2, 3)
    assert ab.flux.unit == units.Jy
    assert np.allclose([[1, 2, 3], [4, 5, 6]], ab.flux.value)

    abb = combine_spectra(ab, b)
    assert isinstance(ab, specutils.Spectrum1D)
    assert abb.shape == (3, 3)
    assert abb.flux.unit == units.Jy
    assert np.allclose([[1, 2, 3], [4, 5, 6], [4, 5, 6]], abb.flux.value)

    c = specutils.Spectrum1D(spectral_axis=[1., 2., 3., 4.]*units.AA,
                             flux=[1., 2., 3., 4.]*units.Jy)

    ac = combine_spectra(a, c)
    assert isinstance(ac, specutils.SpectrumList)

    aca = combine_spectra(ac, a)
    assert isinstance(aca, specutils.SpectrumList)
