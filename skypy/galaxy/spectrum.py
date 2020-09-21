r"""Galaxy spectrum module.

"""

import numpy as np
from astropy import units
from astropy.io.fits import getdata


__all__ = [
    'dirichlet_coefficients',
    'kcorrect_spectra',
    'mag_ab',
]


def dirichlet_coefficients(redshift, alpha0, alpha1, z1=1., weight=None):
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
    weight : (nc,) array_like, optional
        Different weights for each component.

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

    if np.ndim(alpha0) != 1 or np.ndim(alpha1) != 1:
        raise ValueError('alpha0, alpha1 must be 1D arrays')
    if len(alpha0) != len(alpha1):
        raise ValueError('alpha0 and alpha1 must have the same length')
    if weight is not None and (np.ndim(weight) != 1 or len(weight) != len(alpha0)):
        raise ValueError('weight must be 1D and match alpha0, alpha1')

    redshift = np.expand_dims(redshift, -1)

    alpha = np.power(alpha0, 1-redshift/z1)*np.power(alpha1, redshift/z1)

    # sample Dirichlet by normalising independent gamma draws
    coeff = np.random.gamma(alpha)
    if weight is not None:
        coeff *= weight
    coeff /= coeff.sum(axis=-1)[..., np.newaxis]

    return coeff


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


def mag_ab(spectrum, bandpass, redshift=None):
    r'''Compute absolute AB magnitudes from spectra and bandpasses.

    This function takes *emission* spectra and observation bandpasses and
    computes the AB magnitudes for sources at 10pc (i.e. absolute magnitudes).
    The emission spectra can optionally be redshifted. The definition of the
    bandpass AB magnitude is taken from [1]_.

    Both the spectra and the bandpasses must be given as functions of
    wavelength. The spectra must be given as fluxes with units equivalent to
    erg/s/cm2/A while the bandpasses should be dimensionless.

    Parameters
    ----------
    spectrum : specutils.Spectrum1D
        Emission spectra of the sources.
    bandpass : specutils.Spectrum1D
        Bandpass filters.
    redshift : (nz,) array_like, optional
        Optional array of values for redshifting the source spectra.

    Returns
    -------
    mag_ab : (nz, nb, ns) array_like
        Absolute AB magnitudes.

    References
    ----------
    .. [1] M. R. Blanton et al., 2003, AJ, 125, 2348
    '''

    # get the spectra and bandpasses
    spec_lam = spectrum.wavelength.to_value(units.AA, equivalencies=units.spectral())
    spec_flux = spectrum.flux.to_value('erg s-1 cm-2 AA-1',
                                       equivalencies=units.spectral_density(spec_lam))
    band_lam = bandpass.wavelength.to_value(units.AA, equivalencies=units.spectral())
    band_tx = bandpass.flux.to_value(units.dimensionless_unscaled)

    # redshift zero if not given
    if redshift is None:
        redshift = 0.

    # Array shapes
    nz_loop = np.atleast_1d(redshift).shape
    ns_loop = np.atleast_2d(spec_flux).shape[:-1]
    nb_loop = np.atleast_2d(band_tx).shape[:-1]
    nz_return = np.shape(redshift)
    ns_return = spec_flux.shape[:-1]
    nb_return = band_tx.shape[:-1]
    loop_shape = (*nz_loop, *nb_loop, *ns_loop)
    return_shape = (*nz_return, *nb_return, *ns_return)

    # allocate magnitude array
    mag_ab = np.empty(loop_shape, dtype=float)

    # compute magnitude contribution from band normalisation [denominator of (2)]
    m_band = -2.5*np.log10(np.trapz(band_tx/band_lam, band_lam))

    # magnitude offset from band and AB definition [constant in (2)]
    m_offs = -2.4079482426801846 - m_band

    # compute flux integrand at emitted wavelengths
    spec_intg = spec_lam*spec_flux

    # go through redshifts ...
    for i, z in enumerate(np.atleast_1d(redshift)):

        # observed wavelength of spectra
        obs_lam = (1 + z)*spec_lam

        for j, b in enumerate(np.atleast_2d(band_tx)):

            # interpolate band to get transmission at observed wavelengths
            obs_tx = np.interp(obs_lam, band_lam, b, left=0, right=0)

            # compute magnitude contribution from flux [numerator of (2)]
            mag_ab[i, j, :] = -2.5*np.log10(np.trapz(spec_intg*obs_tx, obs_lam))

    # combine AB magnitude [all of (2)]
    mag_ab += np.atleast_1d(m_offs)[:, np.newaxis]

    return mag_ab.item() if not return_shape else mag_ab.reshape(return_shape)
