"""Galaxy redshift module.

This module provides facilities to sample galaxy redshifts using a number of
models.
"""

import numpy as np
import scipy.integrate
import scipy.special
from astropy import units

from ..utils import broadcast_arguments, dependent_argument


__all__ = [
    'redshifts_from_comoving_density',
    'schechter_lf_redshift',
    'schechter_smf_redshift',
    'smail',
]


# largest number x such that exp(x) is a float
_LOGMAX = np.log(np.finfo(0.).max)


def smail(z_median, alpha, beta, size=None):
    r'''Redshifts following the Smail et al. (1994) model.

    The redshift follows the Smail et al. [1]_ redshift distribution.

    Parameters
    ----------
    z_median : float or array_like of floats
        Median redshift of the distribution, must be positive.
    alpha : float or array_like of floats
        Power law exponent (z/z0)^\alpha, must be positive.
    beta : float or array_like of floats
        Log-power law exponent exp[-(z/z0)^\beta], must be positive.
    size : None or int or tuple
        Size of the output. If `None`, the size is inferred from the arguments.
        Default is None.

    Notes
    -----
    The probability distribution function :math:`p(z)` for redshift :math:`z`
    is given by Amara & Refregier [2]_ as

    .. math::

        p(z) \sim \left(\frac{z}{z_0}\right)^\alpha
                    \exp\left[-\left(\frac{z}{z_0}\right)^\beta\right] \;.

    This is the generalised gamma distribution.

    References
    ----------
    .. [1] Smail I., Ellis R. S., Fitchett M. J., 1994, MNRAS, 270, 245
    .. [2] Amara A., Refregier A., 2007, MNRAS, 381, 1018

    '''

    k = (alpha+1)/beta
    t = z_median**beta/scipy.special.gammainccinv(k, 0.5)
    g = np.random.gamma(shape=k, scale=t, size=size)
    return g**(1/beta)


@dependent_argument('M_star', 'redshift')
@dependent_argument('phi_star', 'redshift')
@dependent_argument('alpha', 'redshift')
@broadcast_arguments('redshift', 'M_star', 'phi_star', 'alpha')
@units.quantity_input(sky_area=units.sr)
def schechter_lf_redshift(redshift, M_star, phi_star, alpha, m_lim, sky_area,
                          cosmology, noise=True):
    r'''Sample redshifts from Schechter luminosity function.

    Sample the redshifts of galaxies following a Schechter luminosity function
    with potentially redshift-dependent parameters, limited by an apparent
    magnitude `m_lim`, for a sky area `sky_area`.

    Parameters
    ----------
    redshift : array_like
        Input redshift grid on which the Schechter function parameters are
        evaluated. Galaxies are sampled over this redshift range.
    M_star : array_like or function
        Characteristic absolute magnitude of the Schechter function. Can be a
        single value, an array of values for each `redshift`, or a function of
        redshift.
    phi_star : array_like or function
        Normalisation of the Schechter function. Can be a single value, an
        array of values for each `redshift`, or a function of redshift.
    alpha : array_like or function
        Schechter function power law index. Can be a single value, an array of
        values for each `redshift`, or a function of redshift.
    m_lim : float
        Limiting apparent magnitude.
    sky_area : `~astropy.units.Quantity`
        Sky area over which galaxies are sampled. Must be in units of solid angle.
    cosmology : Cosmology
        Cosmology object to convert apparent to absolute magnitudes.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Returns
    -------
    redshifts : array_like
        Redshifts of the galaxy sample described by the Schechter luminosity
        function.

    '''

    # compute lower truncation of scaled Schechter random variable
    lnxmin = m_lim - cosmology.distmod(np.clip(redshift, 1e-10, None)).value
    lnxmin -= M_star
    lnxmin *= -0.92103403719761827361

    # gamma function integrand
    def f(lnx, a):
        return np.exp((a + 1)*lnx - np.exp(lnx)) if lnx < _LOGMAX else 0.

    # integrate gamma function for each redshift
    gam = np.empty_like(lnxmin)
    for i, _ in np.ndenumerate(gam):
        gam[i], _ = scipy.integrate.quad(f, lnxmin[i], np.inf, args=(alpha[i],))

    # comoving number density is normalisation times upper incomplete gamma
    density = phi_star*gam

    # sample redshifts from the comoving density
    return redshifts_from_comoving_density(redshift=redshift, density=density,
                                           sky_area=sky_area, cosmology=cosmology, noise=noise)


@dependent_argument('m_star', 'redshift')
@dependent_argument('phi_star', 'redshift')
@dependent_argument('alpha', 'redshift')
@broadcast_arguments('redshift', 'm_star', 'phi_star', 'alpha')
@units.quantity_input(sky_area=units.sr)
def schechter_smf_redshift(redshift, m_star, phi_star, alpha, m_min, m_max, sky_area,
                           cosmology, noise=True):
    r'''Sample redshifts from Schechter function.

    Sample the redshifts of galaxies following a Schechter function
    with potentially redshift-dependent parameters, limited by stellar masses
    `m_max` and `m_min`, for a sky area `sky_area`.

    Parameters
    ----------
    redshift : array_like
        Input redshift grid on which the Schechter function parameters are
        evaluated. Galaxies are sampled over this redshift range.
    m_star : array_like or function
        Characteristic stellar mass of the Schechter function. Can be a
        single value, an array of values for each `redshift`, or a function of
        redshift.
    phi_star : array_like or function
        Normalisation of the Schechter function. Can be a single value, an
        array of values for each `redshift`, or a function of redshift.
    alpha : array_like or function
        Schechter function power law index. Can be a single value, an array of
        values for each `redshift`, or a function of redshift.
    m_min : float
        Minimum stellar mass.
    m_max : float
        Maximum stellar mass.
    sky_area : `~astropy.units.Quantity`
        Sky area over which galaxies are sampled. Must be in units of solid angle.
    cosmology : Cosmology
        Cosmology object to convert comoving density.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Returns
    -------
    redshifts : array_like
        Redshifts of the galaxy sample described by the Schechter
        function.

    '''

    lnxmin = np.log(m_min)
    lnxmin -= np.log(m_star)

    lnxmax = np.log(m_max)
    lnxmax -= np.log(m_star)

    # gamma function integrand
    def f(lnx, a):
        return np.exp((a + 1)*lnx - np.exp(lnx)) if lnx < lnxmax.max() else 0.

    # integrate gamma function for each redshift
    gam = np.empty_like(alpha)

    for i, _ in np.ndenumerate(gam):
        gam[i], _ = scipy.integrate.quad(f, lnxmin[i], lnxmax[i], args=(alpha[i],))

    # comoving number density is normalisation times upper incomplete gamma
    density = phi_star*gam

    # sample redshifts from the comoving density
    return redshifts_from_comoving_density(redshift=redshift, density=density,
                                           sky_area=sky_area, cosmology=cosmology, noise=noise)


@units.quantity_input(sky_area=units.sr)
def redshifts_from_comoving_density(redshift, density, sky_area, cosmology, noise=True):
    r'''Sample redshifts from a comoving density function.

    Sample galaxy redshifts such that the resulting distribution matches a past
    lightcone with comoving galaxy number density `density` at redshifts
    `redshift`. The comoving volume sampled corresponds to a sky area `sky_area`
    and transverse comoving distance given by the cosmology `cosmology`.

    If the `noise` parameter is set to true, the number of galaxies has Poisson
    noise. If `noise` is false, the expected number of galaxies is used.

    Parameters
    ----------
    redshift : array_like
        Redshifts at which comoving number densities are provided.
    density : array_like
        Comoving galaxy number density at each redshift in Mpc-3.
    sky_area : `~astropy.units.Quantity`
        Sky area over which galaxies are sampled. Must be in units of solid angle.
    cosmology : Cosmology
        Cosmology object for conversion to comoving volume.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Returns
    -------
    redshifts : array_like
        Sampled redshifts such that the comoving number density of galaxies
        corresponds to the input distribution.

    Warnings
    --------
    The inverse cumulative distribution function is approximated from the
    number density and comoving volume calculated at the given `redshift`
    values. The user must choose suitable `redshift` values to satisfy their
    desired numerical accuracy.

    '''

    # redshift number density
    dN_dz = (cosmology.differential_comoving_volume(redshift) * sky_area).to_value('Mpc3')
    dN_dz *= density

    # integrate density to get expected number of galaxies
    N = np.trapz(dN_dz, redshift)

    # Poisson sample galaxy number if requested
    if noise:
        N = np.random.poisson(N)
    else:
        N = int(N)

    # cumulative trapezoidal rule to get redshift CDF
    cdf = dN_dz  # reuse memory
    np.cumsum((dN_dz[1:]+dN_dz[:-1])/2*np.diff(redshift), out=cdf[1:])
    cdf[0] = 0
    cdf /= cdf[-1]

    # sample N galaxy redshifts
    return np.interp(np.random.rand(N), cdf, redshift)
