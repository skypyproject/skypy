"""Galaxy redshift module.

This module provides facilities to sample galaxy redshifts using a number of
models.


Models
======

.. autosummary::
   :nosignatures:
   :toctree: ../api/

   herbel_redshift
   herbel_pdf
   smail

"""

import numpy as np
import scipy.stats
import scipy.special
import scipy.integrate

from skypy.stats import parametrise, example_args
import skypy.utils.special as special
import skypy.utils.astronomy as astro


@example_args(1.2, 1.5, 2.0)
def smail_par(z_median, alpha, beta):
    r'''Redshifts following the Smail et al (1994) model.

    The redshift follows the Smail et al. [1]_ redshift distribution.

    Parameters
    ----------
    z_median : float or array_like of floats
        Median redshift of the distribution, must be positive.
    alpha : float or array_like of floats
        Power law exponent :math:`(z/z_0)^\alpha`, must be positive.
    beta : float or array_like of floats
        Log-power law exponent :math:`\exp[-(z/z_0)^\beta]`, must be positive.

    Notes
    -----
    The probability distribution function :math:`p(z)` for redshift :math:`z`
    is given by Amara & Refregier [2]_ as

    .. math::

        p(z) \sim \Bigl(\frac{z}{z_0}\Bigr)^\alpha
                    \exp\Bigl[-\Bigl(\frac{z}{z_0}\Bigr)^\beta\Bigr] \;.

    This is the distribution of `scipy.stats.gengamma` with a different
    parametrisation.

    References
    ----------
    .. [1] Smail I., Ellis R. S., Fitchett M. J., 1994, MNRAS, 270, 245
    .. [2] Amara A., Refregier A., 2007, MNRAS, 381, 1018

    %(examples)s
    '''

    a = (alpha+1)/beta
    c = beta
    loc = 0
    scale = z_median*scipy.special.gammainccinv(a, 0.5)**(-1/beta)

    return a, c, loc, scale


smail = parametrise(scipy.stats.gengamma, smail_par, name='smail')


def herbel_redshift(alpha, a_phi, a_m, b_phi, b_m, cosmology, low=0.0,
                    high=2.0,
                    size=None, absolute_magnitude_max=-16.0, resolution=100):
    r""" Redshift following the Schechter luminosity function marginalised over
        luminosities following the Herbel et al. (2017) model.

    Parameters
    ----------
    alpha : float or scalar
        The alpha parameter in the Schechter luminosity function
    a_phi : float or scalar
        Parametrisation factor of the normalisation factor Phi_* as a function
        of redshift according to Herbel et al. (2017) equation (3.4).
    a_m : float or scalar
        Parametrisation factor of the characteristic absolute magnitude M_* as
        a function of redshift according to Herbel et al. (2017) equation (3.3)
    b_phi : float or scalar
        Parametrisation factor of the normalisation factor Phi_* as a function
        of redshift according to Herbel et al. (2017) equation (3.4).
    b_m : float or scalar
        Parametrisation factor of the characteristic absolute magnitude M_* as
        a function of redshift according to Herbel et al. (2017) equation (3.3)
    cosmology : instance
        Instance of an Astropy Cosmology class.
    low : float or array_like of floats, optional
        The lower boundary of teh output interval. All values generated will be
        greater than or equal to low.
        It has to be larger than 0. The default value is 0.01.
    high : float or array_like of floats, optional
        Upper boundary of output interval. All values generated will be less
        than high. The default value is 2.0
    size : int or tuple of ints, optional
        The number of redshifts to sample. If None one value is sampled.
    absolute_magnitude_max : float or scalar, optional
        Upper limit of the considered absolute magnitudes of the galaxies to
        wanna sample.
    resolution : int, optional
        Characterises the resolution of the sampling. Default is 100.

    Returns
    -------
    redshift_sample : ndarray or float
        Drawn redshifts from the marginalised Schechter luminosity function.

    Notes
    -----
    The Schechter luminosity function is given as

    .. math::

        \Phi(L, z) = \frac{\Phi_\star(z)}{L_\star(z)}
            \left(\frac{L}{L_\star(z)}\right)^\alpha
            /exp\left(-\frac{L}{L_\star(z)}\right) \;.

    Here the luminosity is defined as

    .. math::

        L = 10^{-0.4M} \;,

    with absolute magnitude :math:`M`. Furthermore, Herbel et al. (2017)
    introduced

    .. math::

        \Phi_\star(z) = b_\phi \exp(a_\phi z) \;,
        M_\star(z) = a_M z + b_M \;.

    Now we have to rescale the Schechter function by the comoving element and
    get

    .. math::

        \phi(L,z) = \frac{d_H d_M^2}{E(z)}  \Phi(L,z)

    Examples
    --------
    >>> from skypy.galaxy.redshift import herbel_redshift
    >>> from astropy.cosmology import FlatLambdaCDM

    Sample 100 redshift values from the Schechter luminosity function with
    a_m = -0.9408582, b_m = -20.40492365, a_phi = -0.10268436,
    b_phi = 0.00370253, alpha = -1.3.

    >>> cosmology = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    >>> redshift = herbel_redshift(size=1000, low=0.01, alpha=-1.3,
    ...                     a_phi=-0.10268436,a_m=-0.9408582, b_phi=0.00370253,
    ...                     b_m=-20.40492365, cosmology=cosmology,
    ...                     absolute_magnitude_max=-16.)

    References
    ----------
    .. [1] Herbel J., Kacprzak T., Amara A. et al., 2017, Journal of Cosmology
           and Astroparticle Physics, Issue 08, article id. 035 (2017)


    """

    luminosity_min = astro.luminosity_from_absolute_magnitude(
        absolute_magnitude_max)
    redshift = np.linspace(low, high, resolution)
    pdf = herbel_pdf(redshift, alpha, a_phi, a_m, b_phi, b_m, cosmology,
                     luminosity_min)
    cdf = scipy.integrate.cumtrapz(pdf, redshift, initial=0)
    cdf = cdf / cdf[-1]
    u = np.random.uniform(size=size)
    redshift_sample = np.interp(u, cdf, redshift)

    return redshift_sample


def herbel_pdf(redshift, alpha, a_phi, a_m, b_phi, b_m, cosmology,
               luminosity_min):
    """Calculates the redshift pdf of the Schechter luminosity function
    according to the model of Herbel et al. (2017) equation (3.6). That is,
    changing the absolute magnitude M in equation (3.2) to luminosity L,
    integrate over all possible L and multiplying by the comovin element
    using a flat LamdaCDM model to get the corresponding pdf.

    Parameters
    ----------
    redshift : array_like
        Input redshifts.
    alpha : float or scalar
        The alpha parameter in the Schechter luminosity function
    a_phi : float or scalar
        Parametrisation factor of the normalisation factor Phi_* as a function
        of redshift according to Herbel et al. (2017) equation (3.4).
    a_m : float or scalar
        Parametrisation factor of the characteristic absolute magnitude M_* as
        a function of redshift according to Herbel et al. (2017) equation (3.3)
    b_phi : float or scalar
        Parametrisation factor of the normalisation factor Phi_* as a function
        of redshift according to Herbel et al. (2017) equation (3.4).
    b_m : float or scalar
        Parametrisation factor of the characteristic absolute magnitude M_* as
        a function of redshift according to Herbel et al. (2017) equation
        (3.3)
    cosmology : instance
        Instance of an Astropy Cosmology class.
    luminosity_min : float or scalar
        Cut-off luminosity value such that the Schechter luminosity function
        diverges for L -> 0

    Returns
    -------
    ndarray or float
    Un-normalised probability density function as a function of redshift
    according to Herbel et al. (2017)
    """
    dv = cosmology.differential_comoving_volume(redshift).value
    x = luminosity_min \
        * 1./astro.luminosity_from_absolute_magnitude(a_m*redshift + b_m)
    value_gamma = special.upper_incomplete_gamma(alpha + 1, x)
    return dv * b_phi * np.exp(a_phi * redshift) * value_gamma
