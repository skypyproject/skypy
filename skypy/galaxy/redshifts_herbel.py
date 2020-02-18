import numpy as np
from astropy.cosmology import FlatLambdaCDM
import scipy.integrate

import skypy.utils.special as special
import skypy.utils.astronomy as astro


def herbel_redshift(alpha, a_phi, a_m, b_phi, b_m, low=0.0, high=2.0,
                    size=None, absolute_magnitude_max=-16.0):
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
    low : float or array_like of floats
        The lower boundary of teh output interval. All values generated will be
        greater than or equal to low.
        It has to be larger than 0. The default value is 0.01.
    high : float or array_like of floats
        Upper boundary of output interval. All values generated will be less
        than high. The default value is 2.0
    size : int or tuple of ints
        The number of redshifts to sample. If None one value is sampled.
    absolute_magnitude_max : float or scalar.

    Returns
    -------
    redshift_sample : ndarray or float
        Drawn redshifts from the marginalised Schechter luminosity function.

    Notes
    -------
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
    -------
    >>> import skypy.galaxy.redshifts_herbel as herbel

    Sample 100 redshift values from the Schechter luminosity function with
    a_m = -0.9408582, b_m = -20.40492365, a_phi = -0.10268436,
    b_phi = 0.00370253, alpha = -1.3.

    >>> redshifts = herbel.herbel_redshift(size=100, gal_type='blue')

    References
    -------
    [1] Herbel J., Kacprzak T., Amara A. et al., 2017, Journal of Cosmology and
    Astroparticle Physics, Issue 08, article id. 035 (2017)


    """

    luminosity_min = astro.convert_abs_mag_to_lum(absolute_magnitude_max)
    redshift = np.linspace(low, high, 10000)
    pdf = _pdf(redshift, alpha, a_phi, a_m, b_phi, b_m, luminosity_min)
    cdf = scipy.integrate.cumtrapz(pdf, redshift, initial=0)
    cdf = cdf / cdf[-1]
    u = np.random.uniform(size=size)
    redshift_sample = np.interp(u, cdf, redshift)

    return redshift_sample


def _pdf(redshift, alpha, a_phi, a_m, b_phi, b_m, luminosity_min):
    """ Calculates the redshift pdf of the Schechter luminosity function
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
     luminosity_min : float or scalar
         Cut-off luminosity value such that the Schechter luminosity function
         diverges for L -> 0

    Returns
    -------
    ndarray or float
    Un-normalised probability density function as a function of redshift
    according to Herbel et al. (2017)
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    diff_com_el = cosmo.differential_comoving_volume(redshift).value
    return b_phi \
        * np.exp(a_phi * redshift) \
        * special.upper_incomplete_gamma(alpha + 1,
                                         luminosity_min
                                         * 10 ** (0.4 * (
                                                    a_m * redshift + b_m))) \
        * diff_com_el
