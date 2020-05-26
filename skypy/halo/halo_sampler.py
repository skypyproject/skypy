r'''Halo sampler.
This code samples haloes from their mass function.

Models
======
.. autosummary::
   :nosignatures:
   :toctree: ../api/

   Press_Schechter_sampler
   Sheth_Tormen_sampler
'''

import numpy as np
from scipy import integrate
from functools import partial
from astropy.constants import G


__all__ = [
     'halo_mass_function',
     'Press_Schechter_sampler',
     'Sheth_Tormen_sampler',
 ]


def halo_mass_function(A, a, p, M, m_min, m_max, m_star, redshift,
                       power_spectrum, k_min, k_max, cosmology,
                       number=100, resolution=100, size=None):
    """Peak height.
    This function samples haloes from their mass function following Sheth &
    Tormen formalism, see equation 10 in [1]_ or [2]_.

    Parameters
    -----------
    A,a,p: float
        Parameters
    m_min, m_max : array_like
        Lower and upper bounds for the random variable m.
    m_star: float or int
        Factors parameterising the characteristic mass.
    redshift : float
        Redshift value at which to evaluate the variance of the power spectrum.
    power_spectrum: (nk,) array_like
        Linear power spectrum at a single redshift in [Mpc^3].
        The first axis correspond to k values in [Mpc^-1].
    k_min, k_max : float
        Lower and upper bounds for the wavenumbers, in units of [Mpc^-1].
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.
    number : integer, optional
        Number of wavenumber samples to generate, nk. Default is 100.
    resolution : int, optional
        Resolution of the inverse transform sampling spline. Default is 100.
    size: int, optional
        Output shape of samples. Default is None.

    Returns
    --------
    sample: array_like
        Samples drawn from the Sheth-Tormen function, equation (10) in [1].

    Examples
    ---------
    >>> import numpy as np
    >>> import skypy.halo.halo_sampler as hs

    References
    ----------
    .. [1] R. K. Sheth and G. Tormen,  Mon. Not. Roy. Astron. Soc. 308, 119
        (1999), astro-ph/9901122.
    .. [2] https://www.slac.stanford.edu/econf/C070730/talks/
        Wechsler_080207.pdf
    """
    k = np.np.logspace(np.log10(np.min(k_min)), np.log10(np.max(k_max)),
                       num=number)
    Pk = power_spectrum
    Hz = cosmology.H(redshift)

    def sigma_squared(M):
        R = np.power(2 * G.value * M / Hz, 1.0 / 3.0)
        j1 = (np.sin(k * R) - np.cos(k * R) * k * R) / (k * R)
        top_hat = 3 * j1 / (k * R)**2
        integrand = Pk * top_hat**2.0 * k**2.0
        return integrate.simps(integrand, k) / (2. * np.pi ** 2.)

    delta_critical_squared = sigma_squared(m_star)

    m = np.logspace(np.log10(np.min(m_min)), np.log10(np.max(m_max)),
                    resolution)
    nu = delta_critical_squared / sigma_squared(m)

    cdf = _sheth_tormen_cdf(nu, A, a, p)
    t_lower = np.interp(np.min(nu), nu, cdf)
    t_upper = np.interp(np.max(nu), nu, cdf)
    nu_uniform = np.random.uniform(t_lower, t_upper, size=size)
    nu_sample = np.interp(nu_uniform, cdf, nu)

    return nu_sample


def _sheth_tormen_cdf(x, A, a, p):
    PDF = (A / (np.sqrt(np.pi) * x)) * (1.0 + 1.0 / (a * x)**p) *\
        np.sqrt(a * x / 2.0) * np.exp(-(a * x / 2.0))
    CDF = integrate.cumtrapz(PDF, x, initial=0)
    return CDF/CDF[-1]


Press_Schechter_sampler = partial(halo_mass_function, 0.5, 1, 0)
Sheth_Tormen_sampler = partial(halo_mass_function, 0.3222, 0.707, 0.3)
