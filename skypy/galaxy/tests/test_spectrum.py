import numpy as np
import scipy.stats
import pytest
from astropy.io.fits import getdata


try:
    import specutils
except ImportError:
    HAS_SPECUTILS = False
else:
    HAS_SPECUTILS = True


from skypy.galaxy.spectrum import dirichlet_coefficients, kcorrect_spectra


def test_sampling_coefficients():
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


def test_kcorrect_spectra():
    # Download template data
    kcorrect_templates_url = "https://github.com/blanton144/kcorrect/raw/" \
                             "master/data/templates/k_nmf_derived.default.fits"
    lam = getdata(kcorrect_templates_url, 11)
    templates = getdata(kcorrect_templates_url, 1)

    # Test that the shape of the returned flux density corresponds to (nz, nl)
    coefficients = np.array([[0.2, 0.2, 0.2, 0.2, 0.2],
                            [0, 0.1, 0.2, 0.3, 0.4]])
    z = np.array([0.5, 1])
    mass = np.array([5 * 10 ** 10, 7 * 10 ** 9])
    lam_o, sed = kcorrect_spectra(z, mass, coefficients)

    assert sed.shape == (len(z), len(lam))

    # Test that for redshift=0, mass=1 and coefficients=[1,0,0,0,0]
    # the returned wavelengths and spectra match the template data

    coefficients = np.array([1, 0, 0, 0, 0])

    z = np.array([0])
    mass = np.array([1])

    lam_o, sed = kcorrect_spectra(z, mass, coefficients)

    assert np.allclose(lam_o, lam)
    assert np.allclose(sed, templates[0])


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
