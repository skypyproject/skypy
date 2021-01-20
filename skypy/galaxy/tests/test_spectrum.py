import numpy as np
import scipy.stats
import pytest
from skypy.galaxy.spectrum import HAS_SPECLITE


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


@pytest.mark.skipif(not HAS_SPECLITE, reason='test requires speclite')
def test_template_spectra():

    from astropy import units
    from skypy.galaxy.spectrum import mag_ab, SpectrumTemplates
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


@pytest.mark.skipif(not HAS_SPECLITE, reason='test requires speclite')
def test_kcorrect_magnitudes():

    from astropy.cosmology import Planck15
    from skypy.galaxy.spectrum import kcorrect

    # Test returned array shapes with single and multiple filters
    ng, nt = 7, 5
    coeff = np.ones((ng, nt))
    multiple_filters = ['decam2014-g', 'decam2014-r']
    nf = len(multiple_filters)
    z = np.linspace(1, 2, ng)

    MB = kcorrect.absolute_magnitudes(coeff, 'bessell-B')
    assert np.shape(MB) == (ng,)

    MB = kcorrect.absolute_magnitudes(coeff, multiple_filters)
    assert np.shape(MB) == (ng, nf)

    mB = kcorrect.apparent_magnitudes(coeff, z, 'bessell-B', Planck15)
    assert np.shape(mB) == (ng,)

    mB = kcorrect.apparent_magnitudes(coeff, z, multiple_filters, Planck15)
    assert np.shape(mB) == (ng, nf)

    # Test wrong number of coefficients
    nt_bad = 3
    coeff_bad = np.ones((ng, nt_bad))

    with pytest.raises(ValueError):
        MB = kcorrect.absolute_magnitudes(coeff_bad, 'bessell-B')

    with pytest.raises(ValueError):
        MB = kcorrect.absolute_magnitudes(coeff_bad, multiple_filters)

    with pytest.raises(ValueError):
        mB = kcorrect.apparent_magnitudes(coeff_bad, z, 'bessell-B', Planck15)

    with pytest.raises(ValueError):
        mB = kcorrect.apparent_magnitudes(coeff_bad, z, multiple_filters, Planck15)

    # Test stellar_mass parameter
    sm = [10, 20, 30, 40, 50, 60, 70]

    MB = kcorrect.absolute_magnitudes(coeff, 'bessell-B')
    MB_s = kcorrect.absolute_magnitudes(coeff, 'bessell-B', stellar_mass=sm)
    np.testing.assert_allclose(MB_s, MB - 2.5*np.log10(sm))

    MB = kcorrect.absolute_magnitudes(coeff, multiple_filters)
    MB_s = kcorrect.absolute_magnitudes(coeff, multiple_filters, stellar_mass=sm)
    np.testing.assert_allclose(MB_s, MB - 2.5*np.log10(sm)[:, np.newaxis])

    mB = kcorrect.apparent_magnitudes(coeff, z, 'bessell-B', Planck15)
    mB_s = kcorrect.apparent_magnitudes(coeff, z, 'bessell-B', Planck15, stellar_mass=sm)
    np.testing.assert_allclose(mB_s, mB - 2.5*np.log10(sm))

    mB = kcorrect.apparent_magnitudes(coeff, z, multiple_filters, Planck15)
    mB_s = kcorrect.apparent_magnitudes(coeff, z, multiple_filters, Planck15, stellar_mass=sm)
    np.testing.assert_allclose(mB_s, mB - 2.5*np.log10(sm)[:, np.newaxis])


@pytest.mark.skipif(not HAS_SPECLITE, reason='test requires speclite')
def test_kcorrect_stellar_mass():

    from astropy import units
    from skypy.galaxy.spectrum import kcorrect
    from speclite.filters import FilterResponse

    # Gaussian bandpass
    filt_lam = np.logspace(3, 4, 1000) * units.AA
    filt_mean = 5000 * units.AA
    filt_width = 100 * units.AA
    filt_tx = np.exp(-((filt_lam-filt_mean)/filt_width)**2)
    filt_tx[[0, -1]] = 0
    FilterResponse(wavelength=filt_lam, response=filt_tx,
                   meta=dict(group_name='test', band_name='filt'))

    # Using the identity matrix for the coefficients yields trivial test cases
    coeff = np.eye(5)
    Mt = kcorrect.absolute_magnitudes(coeff, 'test-filt')

    # Using the absolute magnitudes of the templates as reference magnitudes
    # should return one solar mass for each template.
    stellar_mass = kcorrect.stellar_mass(coeff, Mt, 'test-filt')
    truth = 1
    np.testing.assert_allclose(stellar_mass, truth)

    # Solution for given magnitudes without template mixing
    Mb = np.array([10, 20, 30, 40, 50])
    stellar_mass = kcorrect.stellar_mass(coeff, Mb, 'test-filt')
    truth = np.power(10, -0.4*(Mb-Mt))
    np.testing.assert_allclose(stellar_mass, truth)
