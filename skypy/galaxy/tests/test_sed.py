import numpy as np
from astropy.io.fits import getdata
from astropy.cosmology import FlatLambdaCDM

from skypy.galaxy.sed import kcorrect_spectra
from skypy.galaxy.spectrum import dirichlet_coefficients


def test_kcorrect_spectra():
    # Test that the shape of the returned flux density corresponds to (nz, nl)
    kcorrect_templates_url = "https://github.com/blanton144/kcorrect/raw/" \
                             "master/data/templates/k_nmf_derived.default.fits"
    lam = getdata(kcorrect_templates_url, 11)
    cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
    alpha0 = np.array([2.079, 3.524, 1.917, 1.992, 2.536])
    alpha1 = np.array([2.265, 3.862, 1.921, 1.685, 2.480])
    weights = np.array([3.47e+09, 3.31e+06, 2.13e+09, 1.64e+10, 1.01e+09])
    z = np.array([0.5, 1])
    mass = np.array([5 * 10 ** 10, 7 * 10 ** 9])
    lam_o, sed = kcorrect_spectra(z, mass, cosmology, alpha0, alpha1, weights)

    assert sed.shape == (len(z), len(lam))

    # Test it returns the right sed and wavelength
    templates = getdata(kcorrect_templates_url, 1)

    d_l = cosmology.luminosity_distance(z).value

    coefficients = dirichlet_coefficients(z, alpha0, alpha1)
    weighted_coeff = np.multiply(coefficients, weights).T.T
    rescaled_coeff = (weighted_coeff.T / weighted_coeff.sum(axis=1) *
                      mass * (10 / (d_l * 10 ** 6)) ** 2).T

    sed_test = (np.matmul(rescaled_coeff, templates).T / (1 + z)).T
    lam_o_test = np.matmul((1 + z).reshape(len(z), 1),
                           lam.reshape(1, len(lam)))

    z = np.array([0.5, 1])
    mass = np.array([5 * 10 ** 10, 7 * 10 ** 9])
    lam_o, sed = kcorrect_spectra(z, mass, cosmology, alpha0, alpha1, weights)

    assert np.allclose(lam_o, lam_o_test)
    assert np.allclose(sed, sed_test)
