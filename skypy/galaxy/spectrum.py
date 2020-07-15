r"""Galaxy spectrum module.

"""

import numpy as np
from astropy.io.fits import getdata


__all__ = [
    'dirichlet_coefficients',
    'kcorrect_spectra'
]


def dirichlet_coefficients(redshift, alpha0, alpha1, z1=1.):
    r"""Dirichlet-distributed SED coefficients.

    Spectral coefficients to calculate the rest-frame spectral energy
        distribution of a galaxy following the Herbel et al. model in [1]_.

    Parameters
    ----------
    redshift : (nz,) array_like
        The redshift values of the galaxies for which the coefficients want to
        be sampled.
    alpha0, alpha1 : (nc,) array_like
        Factors parameterising the Dirichlet distribution according to Equation
        (3.9) in [1]_.
    z1 : float or scalar, optional
       Reference redshift at which alpha = alpha1. The default value is 1.0.

    Returns
    -------
    coefficients : (nz, nc) ndarray
        The spectral coefficients of the galaxies. The shape is (n, nc) with nz
        the number of redshifts and nc the number of coefficients.

    Notes
    -----
    One example is the rest-frame spectral energy distribution of galaxies
    which can be written as a linear combination of the five kcorrect [2]_
    template spectra :math:`f_i`
    (see [1]_)

    .. math::

        f(\lambda) = \sum_{i=1}^5 c_i f_i(\lambda) \;,

    where the coefficients :math:`c_i` were shown to follow a Dirichlet
    distribution of order 5. The five parameters describing the Dirichlet
    distribution are given by

    .. math::

        \alpha_i(z) = (\alpha_{i,0})^{1-z/z_1} \cdot (\alpha_{i,1})^{z/z_1} \;.

    Here, :math:`\alpha_{i,0}` describes the galaxy population at redshift
    :math:`z=0` and :math:`\alpha_{i,1}` the population at :math:`z=z_1 > 0`.
    These parameters depend on the galaxy type and we chose :math:`z_1=1`.

    Beside this example, this code works for a general number of templates.

    References
    ----------
    .. [1] Herbel J., Kacprzak T., Amara A. et al., 2017, Journal of Cosmology
           and Astroparticle Physics, Issue 08, article id. 035 (2017)
    .. [2] Blanton M. R., Roweis S., 2007, The Astronomical Journal,
           Volume 133, Page 734

    Examples
    --------
    >>> from skypy.galaxy.spectrum import dirichlet_coefficients
    >>> import numpy as np

    Sample the coefficients according to [1]_ for n blue galaxies with
    redshifts between 0 and 1.

    >>> n = 100000
    >>> alpha0 = np.array([2.079, 3.524, 1.917, 1.992, 2.536])
    >>> alpha1 = np.array([2.265, 3.862, 1.921, 1.685, 2.480])
    >>> redshift = np.linspace(0,2, n)
    >>> coefficients = dirichlet_coefficients(redshift, alpha0, alpha1)

    """
    if np.isscalar(alpha0) or np.isscalar(alpha1):
        raise ValueError("alpha0 and alpha1 must be array_like.")
    return_shape = (*np.shape(redshift), *np.shape(alpha0))
    redshift = np.atleast_1d(redshift)[:, np.newaxis]

    alpha = np.power(alpha0, 1. - redshift / z1) * \
        np.power(alpha1, redshift / z1)

    # To sample Dirichlet distributed variables of order k we first sample k
    # Gamma distributed variables y_i with the parameters alpha_i.
    # Normalising all y_i by the sum of all k y_i gives us the k Dirichlet
    # distributed variables.
    y = np.random.gamma(alpha)
    sum_y = y.sum(axis=1)
    coefficients = np.divide(y.T, sum_y.T).T
    return coefficients.reshape(return_shape)


def kcorrect_spectra(redshift, stellar_mass, coefficients):
    r"""Flux densities of galaxies.

    The flux density as a sum of the 5 kcorrect templates.

    Parameters
    ----------
    redshift : (nz,) array-like
        The redshift values of the galaxies.
    stellar_mass : (nz, ) array-like
        The stellar masses of the galaxies.
    coefficients: (nz, 5) array-like
        Coefficients to be multiplied with the kcorrect templates.


    Returns
    -------
    wavelength_observe : (nl, ) array_like
        Wavelengths corresponding to the flux density. Given in units of
        Angstrom
    sed: (nz, nl) array-like
        Flux density of the galaxies in units of erg/s/cm^2/Angstrom as it
        would be observed at a distance of 10 pc.

    Notes
    -----
    The rest-frame flux-density can be calculated as a sum of the five kcorrect
    templates [1]_

    .. math::
        f_e(\lambda) = \sum_i c_i t_i(\lambda) \;,

    with kcorrect templates :math:`t_i(\lambda)` and coefficients :math:`c_i`.

    The kcorrect templates are given in units of
    erg/s/cm^2/Angstrom per solar mass and as it would be observed in a
    distance of 10pc. To obtain the correct flux density if the object would be
    at 10 pc distance we have to adjust the coefficients by the stellar mass
    :math:`M` of the galaxy

    .. math::
         \tilde{c_i} = c_i \cdot M \;.

    Thus, the flux density is given by

    .. math::
        f_e(\lambda) = \sum_i \tilde{c_i} t_i(\lambda) \;.

    To get the flux density in observed frame we have to redshift it

    .. math::
        f_o(\lambda_o) = \frac{f_e(\lambda)}{1+z} \;

    where

    .. math::
        \lambda_o = (1+z) \lambda \;.

    References
    ----------
    .. [1] Blanton M., Roweis S., 2006, The Astronomical Journal, Issue 2,
        Volume 133, Pages 734 - 754

    Examples
    --------
    >>> from skypy.galaxy.spectrum import kcorrect_spectra
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> from skypy.galaxy.spectrum import dirichlet_coefficients

    Calculate the flux density for two galaxies.

    >>> cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
    >>> alpha0 = np.array([2.079, 3.524, 1.917, 1.992, 2.536])
    >>> alpha1 = np.array([2.265, 3.862, 1.921, 1.685, 2.480])
    >>> redshift = np.array([0.5,1])
    >>> coefficients = dirichlet_coefficients(redshift, alpha0, alpha1)
    >>> stellar_mass = np.array([5*10**10, 7*10**9])
    >>>
    >>> wavelength_o, sed = kcorrect_spectra(redshift, stellar_mass,
    ...                                       coefficients)

    """

    kcorrect_templates_url = "https://github.com/blanton144/kcorrect/raw/" \
                             "master/data/templates/k_nmf_derived.default.fits"
    templates = getdata(kcorrect_templates_url, 1)
    wavelength = getdata(kcorrect_templates_url, 11)

    rescaled_coeff = (coefficients.T * stellar_mass).T

    sed = (np.matmul(rescaled_coeff, templates).T / (1 + redshift)).T
    wavelength_observed = np.matmul((1 + redshift).reshape(len(redshift), 1),
                                    wavelength.reshape(1, len(wavelength)))

    return wavelength_observed, sed
