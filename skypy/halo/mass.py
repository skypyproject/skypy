r'''Halo mass sampler.
This code samples halos from their mass function.

Models
======
.. autosummary::
   :nosignatures:
   :toctree: ../api/

   press_schechter
   Press_Schechter_sampler
   Sheth_Tormen_sampler
'''

import numpy as np
from scipy import integrate
from functools import partial
from astropy.constants import G

from skypy.utils.random import schechter

__all__ = [
     'press_schechter',
     'halo_mass_function',
     'Press_Schechter_sampler',
     'Sheth_Tormen_sampler',
 ]


def press_schechter(n, m_star, size=None, x_min=0.00305,
                    x_max=1100.0, resolution=100):

    """Sampling from Press-Schechter mass function (1974).

    Masses following the Press-Schechter mass function following the
    Press et al. [1]_ formalism.
    Parameters
    ----------
    n : float or int
        The n parameter in the Press-Schechter mass function.
    m_star : float or int
        Factors parameterising the characteristic mass.
    size: int, optional
         Output shape of luminosity samples.
    x_min, x_max : float or int, optional
        Lower and upper bounds in units of M*.
    resolution : int, optional
        Resolution of the inverse transform sampling spline. Default is 100.
    Returns
    -------
    mass : array_like
        Drawn masses from the Press-Schechter mass function.

    Examples
    --------
    >>> import skypy.halo.mass as mass
    >>> n, m_star = 1, 1e9
    >>> sample = mass.press_schechter(n, m_star, size=1000, x_min=1e-10,
    ...                               x_max=1e2, resolution=1000)

    References
    ----------
    .. [1] Press, W. H. and Schechter, P., APJ, (1974).
    """

    alpha = - 0.5 * (n + 9.0) / (n + 3.0)

    x_sample = schechter(alpha, x_min, x_max, resolution=resolution, size=size)

    return m_star * np.power(x_sample, 3.0 / (n + 3.0))


def halo_mass_function(A, a, p, m_min, m_max, m_star, redshift,
                       power_spectrum, wavenumber, cosmology,
                       resolution=100, size=None):
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
    wavenumber : (nk,) array_like
        Array of wavenumbers at which the power spectrum is evaluated,
        in units of [Mpc^-1].
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.
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
    k = wavenumber
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
