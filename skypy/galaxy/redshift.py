"""Galaxy redshift module.

This module provides facilities to sample galaxy redshifts using a number of
models.
"""

import numpy as np
import scipy.integrate
import scipy.special

from ..utils import uses_default_cosmology, broadcast_arguments, dependent_argument


__all__ = [
    'redshifts_from_comoving_density',
    'schechter_lf_redshift',
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

    Examples
    --------
    Sample 10 random variates from the Smail model with `alpha = 1.5` and
    `beta = 2` and median redshift `z_median = 1.2`.

    >>> from skypy.galaxy.redshift import smail
    >>> redshift = smail(1.2, 1.5, 2.0, size=10)

    '''

    k = (alpha+1)/beta
    t = z_median**beta/scipy.special.gammainccinv(k, 0.5)
    g = np.random.gamma(shape=k, scale=t, size=size)
    return g**(1/beta)


@uses_default_cosmology
@dependent_argument('M_star', 'redshift')
@dependent_argument('phi_star', 'redshift')
@dependent_argument('alpha', 'redshift')
@broadcast_arguments('redshift', 'M_star', 'phi_star', 'alpha')
def schechter_lf_redshift(redshift, M_star, phi_star, alpha, m_lim, fsky, cosmology, noise=True):
    r'''Sample redshifts from Schechter luminosity function.

    Sample the redshifts of galaxies following a Schechter luminosity function
    with potentially redshift-dependent parameters, limited by an apparent
    magnitude `m_lim`, for a fraction `fsky` of the sky.

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
    fsky : array_like
        Sky fraction over which galaxies are sampled.
    cosmology : Cosmology, optional
        Cosmology object to convert apparent to absolute magnitudes. If not
        given, the default cosmology is used.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Returns
    -------
    redshifts : array_like
        Redshifts of the galaxy sample described by the Schechter luminosity
        function.

    Examples
    --------
    Compute the number density of galaxies with redshifts between 0 and 5
    for typical values of the "blue" galaxy luminosity function above an
    apparent magnitude cut of 22 for a survey of 1 square degree = 1/41253 of
    the sky.

    >>> from skypy.galaxy.redshift import schechter_lf_redshift
    >>> z = [0., 5.]
    >>> M_star = -20.5
    >>> phi_star = 3.5e-3
    >>> alpha = -1.3
    >>> z_gal = schechter_lf_redshift(z, M_star, phi_star, alpha, 22, 1/41253)

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
                                           fsky=fsky, cosmology=cosmology, noise=noise)


@uses_default_cosmology
def redshifts_from_comoving_density(redshift, density, fsky, cosmology, noise=True):
    r'''Sample redshifts from a comoving density function.

    Sample galaxy redshifts such that the resulting distribution matches a past
    lightcone with comoving galaxy number density `density` at redshifts
    `redshift`. The comoving volume sampled corresponds to a sky fraction `fsky`
    and transverse comoving distance given by the cosmology `cosmology`.

    If the `noise` parameter is set to true, the number of galaxies has Poisson
    noise. If `noise` is false, the expected number of galaxies is used.

    Parameters
    ----------
    redshift : array_like
        Redshifts at which comoving number densities are provided.
    density : array_like
        Comoving galaxy number density at each redshift in Mpc-3.
    fsky : array_like
        Sky fraction over which galaxies are sampled.
    cosmology : Cosmology, optional
        Cosmology object for conversion to comoving volume. If not given, the
        default cosmology is used.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Returns
    -------
    redshifts : array_like
        Sampled redshifts such that the comoving number density of galaxies
        corresponds to the input distribution.

    Examples
    --------
    Sample redshifts with a constant comoving number density 1e-3/Mpc3 up to
    redshift 1 for a survey of 1 square degree = 1/41253 of the sky.

    >>> from skypy.galaxy.redshift import redshifts_from_comoving_density
    >>> z_range = np.arange(0, 1.01, 0.1)
    >>> z_gal = redshifts_from_comoving_density(z_range, 1e-3, 1/41253)

    '''

    # redshift number density
    dN_dz = cosmology.differential_comoving_volume(redshift).to_value('Mpc3/sr')
    dN_dz *= density
    dN_dz *= 4*np.pi*fsky

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
