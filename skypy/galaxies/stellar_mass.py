"""Models of galaxy stellar mass.
"""

import numpy as np
from scipy.integrate import trapz

from ..utils.random import schechter
from ..utils import dependent_argument


__all__ = [
    'schechter_smf_mass',
    'schechter_smf_phi_active_redshift',
]


@dependent_argument('m_star', 'redshift')
@dependent_argument('alpha', 'redshift')
def schechter_smf_mass(redshift, alpha, m_star, m_min, m_max, size=None,
                       resolution=1000):
    r""" Stellar masses following the Schechter mass function [1]_.

    Parameters
    ----------
    redshift : array_like
        Galaxy redshifts for which to sample magnitudes.
    alpha : float or function
        The alpha parameter in the Schechter stellar mass function. If function,
        it must return a scalar value.
    m_star : (nm,) array-like or function
        Characteristic stellar mass m_*.
    size: int, optional
         Output shape of stellar mass samples. If size is None and m_star
         is a scalar, a single sample is returned. If size is None and
         m_star is an array, an array of samples is returned with the same
         shape as m_star.
    m_min, m_max : float
        Lower and upper bounds for the stellar mass.
    resolution : int, optional
        Resolution of the inverse transform sampling spline. Default is 100.

    Returns
    -------
    stellar mass : (nm,) array_like
        Drawn stellar masses from the Schechter stellar mass function in units
        of the solar mass.

    Notes
    -----
    The stellar mass probability distribution (pdf) follows a Schechter
    profile of the form

    .. math::

        \Phi(M) = \frac{1}{M_*} \left(\frac{M}{M_*}\right)^\alpha
            \exp\left(-\frac{M}{M_*}\right) \;.

    From this pdf one can sample the stellar masses.

    References
    ----------
    .. [1] Mo, H., Van den Bosch, F., & White, S. (2010). Galaxy Formation and
        Evolution. Cambridge: Cambridge University Press.
        doi:10.1017/CBO9780511807244

    """

    # only alpha scalars supported at the moment
    if np.ndim(alpha) > 0:
        raise NotImplementedError('only scalar alpha is supported')

    if size is None and np.shape(redshift):
        size = np.shape(redshift)

    # convert m_min, m_max to units of m_star
    x_min = m_min / m_star
    x_max = m_max / m_star

    # sample masses
    m = schechter(alpha, x_min, x_max, resolution, size=size, scale=m_star)

    return m


def schechter_smf_phi_active_redshift(redshift,
                                      phi_active_today,
                                      probability_satellites,
                                      cosmology,
                                      redshift_initial=10,
                                      resolution=100):
    r'''Time-dependent Schechter amplitude of active galaxies.
    This function returns the time-dependent Schechter mass function amplitude
    for the total active population based on equation (17)
    in de la Bella et al. 2021 [1]_.

    Parameters
    ----------
    redshift:
        Values of redshift at which to evaluate the amplitude.

    phi_active_today: array_like
        Schechter mass function amplitude for the entire active
        sample of galaxies in the past when density and mass was very low.

    probability_satellites: func
        Probability of active galaxies becoming satellites
        as a function of redshift.

    cosmology: astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history
        of omega_matter and omega_lambda with redshift.

    redshift_initial: float
        Value of redshift in the past when density and mass was very low.
        Default is 10.

    resolution: float
        Resolution of the integral. Default is 100.

    Returns
    -------
    amplitude: array_like
        Amplitude of the Schechter mass function.

     References
    ----------
    .. [1] de la Bella et al. 2021, Quenching and Galaxy Demographics,
           arXiv 2112.11110.

    '''

    def integrand(z):
        return probability_satellites(z) / cosmology.H(z).value / (1 + z)

    # Calculate the amplitude in the past
    redshift_today = np.linspace(0, redshift_initial, resolution)
    integrand_today = integrand(redshift_today)
    integral_today = trapz(integrand_today)
    B = phi_active_today * np.exp(- integral_today) / (1 - integral_today)

    # Calculate the amplitude for the given redshift
    redshift_array = np.linspace(redshift, redshift_initial, resolution)
    integrand_redshift = integrand(redshift_array)
    integral = trapz(integrand_redshift)

    return B * (1 - integral) * np.exp(integral)
