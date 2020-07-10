r"""Models of galaxy spectra.

"""


import numpy as np
from astropy.io.fits import getdata

from skypy.galaxy.spectrum import dirichlet_coefficients


__all__ = [
    'kcorrect_spectra'
]


def kcorrect_spectra(redshift, stellar_mass, cosmology, alpha0, alpha1,
                     weights):
    r"""Flux densities of galaxies.

    The flux density as a sum of the 5 kcorrect templates [1]_.

    Parameters
    ----------
    redshift : (nz,) array-like
        The redshift values of the galaxies.
    stellar_mass : (nz, ) array-like
        The stellar masses of the galaxies.
    cosmology : instance
        Instance of an Astropy Cosmology class.
    alpha0, alpha1: (5, ) array-like
        Redshift parametrisation of the Dirichlet parameters.
    weights : (5, ) array-like
        Weighting of the Dirichlet parameters

    Returns
    -------
    wavelength_observe : (nl, ) array_like
        Wavelengths corresponding to the flux density. Given in units of
        Angstrom
    sed: (nz, nl) array-like
        Flux density of the galaxies in units of erg/s/cm^2/Angstrom

    Notes
    -----
    The rest-frame flux-density can be calculated as a sum of the five kcorrect
    templates ([1]_, [2]_)

    .. math::
        f_e(\lambda) = \sum_i c_i t_i(\lambda) \;,

    with kcorrect templates :math:`t_i(\lambda)` and coefficients :math:`c_i`.
    [2]_ showed that weighting the coefficients :math:`c_i` they follow a
    Dirichlet distribution of order five.
    They showed as well that the Dirichlet parameters :math:`alpha_i` are
    redshift dependent and are given as

    .. math::
        \alpha_i(z) = (\alpha_{i,0})^{1-z/z_1} \cdot (\alpha_{i,1})^{z/z_1} \;.

    Here :math:`\alpha_{i,0}` describes the galaxy distribution at redshift
    :math:`z = 0` and  :math:`\alpha_{i,1}` at :math:`z = z_1 > 0`.

    If we follow this approach to draw the coefficients we have to apply the
    weights :math:`w_i' introduced by [2]_ and rescale the coefficients such
    that their sum is again one

    .. math::
    \tilde{c_i} = \frac{c_i \cdot w_i}{\sum_i w_i} \;.

    Furthermore, the kcorrect templates are given in units of
    erg/s/cm^2/Angstrom per solar mass and as it would be observed in a
    distance of 10pc. To obtain the correct flux density we such have to adjust
    the coefficients by the stellar mass :math:`M` of the galaxy and it's
    luminosity distance :math:`d_l`

    .. math::
         \tilde{c_i}' = \tilde{c_i} \cdot M \cdot
         \left(\frac{10\,\mathrm{pc}}{d_l}\right)^2 \;.

    At the end, the flux density is given by

    .. math::
        f_e(\lambda) = \sum_i \tilde{c_i}' t_i(\lambda) \;,

    To get the flux density in observed frame we have to redshift it

    .. math::
        f_o(\lambda_o) = \frac{f_e(\lambda)}{1+z} \;.



    References
    ----------
    .. [1] Blanton M., Roweis S., 2006, The Astronomical Journal, Issue 2,
        Volume 133, Pages 734 - 754
    .. [2] Herbel J., Kacprzak T., Amara A. et al., 2017, Journal of Cosmology
        and Astroparticle Physics, Issue 08, article id. 035 (2017)

    Examples
    --------
    >>> from skypy.galaxy.sed import kcorrect_spectra
    >>> from astropy.cosmology import FlatLambdaCDM

    Calculate the flux density for two galaxies.

    >>> cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
    >>> alpha0 = np.array([2.079, 3.524, 1.917, 1.992, 2.536])
    >>> alpha1 = np.array([2.265, 3.862, 1.921, 1.685, 2.480])
    >>> weights = np.array([3.47e+09, 3.31e+06, 2.13e+09, 1.64e+10, 1.01e+09])
    >>> redshift = np.array([0.5,1])
    >>> stellar_mass = np.array([5*10**10, 7*10**9])
    >>>
    >>> wavelength_o, sed = kcorrect_spectra(redshift, stellar_mass, cosmology,
    ...                                       alpha0, alpha1, weights)

    """

    kcorrect_templates_url = "https://github.com/blanton144/kcorrect/raw/" \
                             "master/data/templates/k_nmf_derived.default.fits"
    templates = getdata(kcorrect_templates_url, 1)
    wavelength = getdata(kcorrect_templates_url, 11)

    luminosity_distance = cosmology.luminosity_distance(redshift).value

    coefficients = dirichlet_coefficients(redshift, alpha0, alpha1)
    weighted_coeff = np.multiply(coefficients, weights).T.T
    rescaled_coeff = (weighted_coeff.T / weighted_coeff.sum(axis=1) *
                      stellar_mass * (10/(luminosity_distance * 10**6))**2).T

    sed = (np.matmul(rescaled_coeff, templates).T / (1 + redshift)).T
    wavelength_observed = np.matmul((1 + redshift).reshape(len(redshift), 1),
                                    wavelength.reshape(1, len(wavelength)))

    return wavelength_observed, sed
