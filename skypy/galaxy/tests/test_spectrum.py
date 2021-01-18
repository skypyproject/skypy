import numpy as np
import scipy.stats
import pytest
from skypy.galaxy.spectrum import HAS_SPECUTILS, HAS_SPECLITE


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


@pytest.mark.skipif(not HAS_SPECLITE, reason='test requires speclite')
def test_mag_ab_standard_source():

    from astropy import units
    from speclite.filters import FilterResponse
    from skypy.galaxy.spectrum import mag_ab

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
    from skypy.galaxy.spectrum import mag_ab

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
    from skypy.galaxy.spectrum import mag_ab
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


@pytest.mark.skipif(not HAS_SPECUTILS or not HAS_SPECLITE,
                    reason='test requires specutils and speclite')
def test_template_spectra():

    from astropy import units
    from skypy.galaxy.spectrum import mag_ab
    from astropy.cosmology import Planck15
    from speclite.filters import FilterResponse

    # 3 Flat Templates
    lam = np.logspace(-1, 4, 1000)*units.AA
    A = np.array([[2], [3], [4]])
    flam = A * 0.10885464149979998*units.Unit('erg s-1 cm-2 AA')/lam**2

    # Gaussian bandpass
    filt_lam = np.logspace(0, 4, 1000)*units.AA
    filt_tx = np.exp(-((filt_lam - 1000*units.AA)/(100*units.AA))**2)
    filt_tx[[0, -1]] = 0
    FilterResponse(wavelength=filt_lam, response=filt_tx,
                   meta=dict(group_name='test', band_name='filt'))

    # Each test galaxy is exactly one of the templates
    coefficients = np.eye(3)
    mt = mag_ab(lam, flam, 'test-filt', coefficients=coefficients)
    m = mag_ab(lam, flam, 'test-filt')
    np.testing.assert_allclose(mt, m)

    # Test distance modulus
    redshift = np.array([1, 2, 3])
    dm = Planck15.distmod(redshift).value
    mt = mag_ab(lam, flam, 'test-filt', coefficients=coefficients, distmod=dm)
    np.testing.assert_allclose(mt, m + dm)

    # Redshift interpolation test; linear interpolation sufficient over a small
    # redshift range at low relative tolerance
    z = np.linspace(0.1, 0.2, 3)
    m_true = mag_ab(lam, flam, 'test-filt', redshift=z, coefficients=coefficients, interpolate=4)
    m_interp = mag_ab(lam, flam, 'test-filt', redshift=z, coefficients=coefficients, interpolate=2)
    np.testing.assert_allclose(m_true, m_interp, rtol=1e-5)
    assert not np.all(m_true == m_interp)


@pytest.mark.skipif(not HAS_SPECLITE, reason='test requires speclite')
def test_kcorrect_magnitudes():

    from astropy.cosmology import Planck15
    from skypy.galaxy.spectrum import kcorrect_absolute_magnitudes, kcorrect_apparent_magnitudes

    # Test returned array shapes with single and multiple filters
    ng, nt = 7, 5
    coeff = np.ones((ng, nt))
    multiple_filters = ['decam2014-g', 'decam2014-r']
    nf = len(multiple_filters)
    z = np.linspace(1, 2, ng)

    MB = kcorrect_absolute_magnitudes(coeff, 'bessell-B')
    assert np.shape(MB) == (ng,)

    MB = kcorrect_absolute_magnitudes(coeff, multiple_filters)
    assert np.shape(MB) == (ng, nf)

    mB = kcorrect_apparent_magnitudes(coeff, z, 'bessell-B', Planck15)
    assert np.shape(mB) == (ng,)

    mB = kcorrect_apparent_magnitudes(coeff, z, multiple_filters, Planck15)
    assert np.shape(mB) == (ng, nf)

    # Test wrong number of coefficients
    nt_bad = 3
    coeff_bad = np.ones((ng, nt_bad))

    with pytest.raises(ValueError):
        MB = kcorrect_absolute_magnitudes(coeff_bad, 'bessell-B')

    with pytest.raises(ValueError):
        MB = kcorrect_absolute_magnitudes(coeff_bad, multiple_filters)

    with pytest.raises(ValueError):
        mB = kcorrect_apparent_magnitudes(coeff_bad, z, 'bessell-B', Planck15)

    with pytest.raises(ValueError):
        mB = kcorrect_apparent_magnitudes(coeff_bad, z, multiple_filters, Planck15)

    # Test stellar_mass parameter
    sm = [10, 20, 30, 40, 50, 60, 70]

    MB = kcorrect_absolute_magnitudes(coeff, 'bessell-B')
    MB_s = kcorrect_absolute_magnitudes(coeff, 'bessell-B', stellar_mass=sm)
    np.testing.assert_allclose(MB_s, MB - 2.5*np.log10(sm))

    MB = kcorrect_absolute_magnitudes(coeff, multiple_filters)
    MB_s = kcorrect_absolute_magnitudes(coeff, multiple_filters, stellar_mass=sm)
    np.testing.assert_allclose(MB_s, MB - 2.5*np.log10(sm)[:, np.newaxis])

    mB = kcorrect_apparent_magnitudes(coeff, z, 'bessell-B', Planck15)
    mB_s = kcorrect_apparent_magnitudes(coeff, z, 'bessell-B', Planck15, stellar_mass=sm)
    np.testing.assert_allclose(mB_s, mB - 2.5*np.log10(sm))

    mB = kcorrect_apparent_magnitudes(coeff, z, multiple_filters, Planck15)
    mB_s = kcorrect_apparent_magnitudes(coeff, z, multiple_filters, Planck15, stellar_mass=sm)
    np.testing.assert_allclose(mB_s, mB - 2.5*np.log10(sm)[:, np.newaxis])


@pytest.mark.skipif(not HAS_SPECUTILS or not HAS_SPECLITE,
                    reason='test requires specutils and speclite')
def test_kcorrect_stellar_mass():

    from astropy import units
    from astropy.io import fits
    from pkg_resources import resource_filename
    from skypy.galaxy.spectrum import mag_ab, kcorrect_stellar_mass
    from speclite.filters import FilterResponse

    # Gaussian bandpass
    filt_lam = np.logspace(3, 4, 1000) * units.AA
    filt_mean = 5000 * units.AA
    filt_width = 100 * units.AA
    filt_tx = np.exp(-((filt_lam-filt_mean)/filt_width)**2)
    filt_tx[[0, -1]] = 0
    FilterResponse(wavelength=filt_lam, response=filt_tx,
                   meta=dict(group_name='test', band_name='filt'))

    # Absolute magnitudes for each kcorrect template
    filename = resource_filename('skypy', 'data/kcorrect/k_nmf_derived.default.fits')
    with fits.open(filename) as hdul:
        flam = hdul[1].data * units.Unit('erg s-1 cm-2 angstrom-1')
        lam = hdul[11].data * units.Unit('angstrom')
    Mt = mag_ab(lam, flam, 'test-filt')

    # Using the identity matrix for the coefficients yields trivial test cases
    coeff = np.eye(5)

    # Using the absolute magnitudes of the templates as reference magnitudes
    # should return one solar mass for each template.
    stellar_mass = kcorrect_stellar_mass(coeff, Mt, 'test-filt')
    truth = 1
    np.testing.assert_allclose(stellar_mass, truth)

    # Solution for given magnitudes without template mixing
    Mb = np.array([10, 20, 30, 40, 50])
    stellar_mass = kcorrect_stellar_mass(coeff, Mb, 'test-filt')
    truth = np.power(10, -0.4*(Mb-Mt))
    np.testing.assert_allclose(stellar_mass, truth)


@pytest.mark.skipif(not HAS_SPECUTILS or not HAS_SPECLITE,
                    reason='test requires specutils and speclite')
def test_load_spectral_data():

    from skypy.galaxy.spectrum import load_spectral_data
    from astropy.utils.data import get_pkg_data_filename

    # load a local file
    filename = get_pkg_data_filename('data/spectrum.ecsv')
    load_spectral_data(filename)

    # load skypy data spectrum templates
    load_spectral_data('kcorrect_spec')

    # Load speclite bandpasses
    load_spectral_data('decam2014_ugrizY')
    load_spectral_data('sdss2010_ugriz')
    load_spectral_data('wise2010_W1W2W3W4')
    load_spectral_data('bessell_UBVRI')

    # load multiple sources
    load_spectral_data(['BASS_gr', 'MzLS_z'])

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
    from specutils import Spectrum1D, SpectrumList

    a = Spectrum1D(spectral_axis=[1., 2., 3.]*units.AA, flux=[1., 2., 3.]*units.Jy)
    b = Spectrum1D(spectral_axis=[1e-10, 2e-10, 3e-10]*units.m,
                   flux=[4e-23, 5e-23, 6e-23]*units.Unit('erg s-1 cm-2 Hz-1'))

    assert np.allclose(a.spectral_axis, b.spectral_axis, atol=0, rtol=1e-10)

    assert a == combine_spectra(a, None)
    assert a == combine_spectra(None, a)

    ab = combine_spectra(a, b)
    assert isinstance(ab, Spectrum1D)
    assert ab.shape == (2, 3)
    assert ab.flux.unit == units.Jy
    assert np.allclose([[1, 2, 3], [4, 5, 6]], ab.flux.value)

    abb = combine_spectra(ab, b)
    assert isinstance(ab, Spectrum1D)
    assert abb.shape == (3, 3)
    assert abb.flux.unit == units.Jy
    assert np.allclose([[1, 2, 3], [4, 5, 6], [4, 5, 6]], abb.flux.value)

    c = Spectrum1D(spectral_axis=[1., 2., 3., 4.]*units.AA, flux=[1., 2., 3., 4.]*units.Jy)

    ac = combine_spectra(a, c)
    assert isinstance(ac, SpectrumList)

    aca = combine_spectra(ac, a)
    assert isinstance(aca, SpectrumList)
