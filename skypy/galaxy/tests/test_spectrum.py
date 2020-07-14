import numpy as np
import scipy.stats
import pytest
from astropy.io.fits import getdata


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
    with pytest.raises(ValueError,
                       match="alpha0 and alpha1 must be array_like."):
        dirichlet_coefficients(redshift, scalar_alpha, alpha1)
    with pytest.raises(ValueError,
                       match="alpha0 and alpha1 must be array_like."):
        dirichlet_coefficients(redshift, alpha0, scalar_alpha)


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
