r'''Halo mass sampler.
This code samples halos from their mass function.

Models
======
.. autosummary::
   :nosignatures:
   :toctree: ../api/

   press_schechter
   halo_mass_function
   halo_mass_sampler
   sheth_tormen_collapse_function
   press_schechter_collapse_function
'''

import numpy as np
from scipy import integrate
from scipy.misc import derivative
from functools import partial
from astropy.constants import G
from astropy import units as u

from skypy.power_spectrum import growth_function
from skypy.utils.random import schechter

__all__ = [
    'press_schechter',
    'halo_mass_function',
    'halo_mass_sampler',
    'sheth_tormen_collapse_function',
    'press_schechter_collapse_function',
 ]


def _sigma_squared(M, wavenumber, linear_power_today, cosmology, redshift):
    k = wavenumber
    Pk = linear_power_today
    Hz = cosmology.H(redshift).value
    if isinstance(M, np.ndarray):
        M = M[:, np.newaxis]

    # The linear frowth function
    D0 = growth_function(0, cosmology)
    Dz = growth_function(redshift, cosmology) / D0

    R = np.power(2 * G.value * M / Hz, 1.0 / 3.0)
    j1 = (np.sin(k * R) - np.cos(k * R) * k * R) / (k * R)
    top_hat = 3 * j1 / (k * R)**2
    integrand = Pk * top_hat**2.0 * k**2.0
    return Dz**2 * integrate.simps(integrand, k) / (2. * np.pi ** 2.)


def halo_mass_function(collapse_function, m_min, m_max, delta_c, redshift,
                       wavenumber, power_spectrum, cosmology,
                       resolution=100, step=1.0e-6):
    r'''Halo mass function.
    This function computes the halo mass function, defined
    in equation 7.46 in [1]_.

    Parameters
    -----------
    collapse_function: (nm,) array_like
        Collapse function to choose from a variety of models:
        `sheth_tormen_collapse_function`, `press_schechter_collapse_function`.
    m_min, m_max : float
        Lower and upper bounds for the random variable `m` in solar mass.
    delta_c: float
        Critical density for collapsed objects.
    redshift : float
        Redshift value at which to evaluate the variance of the power spectrum.
    wavenumber : (nk,) array_like
        Array of wavenumbers at which the power spectrum is evaluated,
        in units of :math:`[Mpc^-1]`.
    power_spectrum: (nk,) array_like
        Linear power spectrum at redshift 0 in :math:`[Mpc^3]`.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.
    resolution : int, optional
        Resolution of the inverse transform sampling spline. Default is 100.
    step : float
        Step size used in the derivative. Default is 1.0e-6.

    Returns
    --------
    mass_function: (nm,) array_like
        Halo mass function for a given mass array, cosmology and redshift.

    Examples
    ---------
    >>> import numpy as np
    >>> from skypy.halo import mass
    >>> from skypy.power_spectrum import _eisenstein_hu as eh

    This example will compute the halo mass function for a
    Planck15 cosmology

    >>> from astropy.cosmology import Planck15
    >>> cosmology = Planck15
    >>> m_min, m_max = 0.1, 10.0
    >>> m = np.logspace(np.log10(m_min), np.log10(m_max), num=4)
    >>> k = np.logspace(-3, 1, num=5, base=10.0)
    >>> A_s, n_s = 2.1982e-09, 0.969453
    >>> Pk = eh.eisenstein_hu(k, A_s, n_s, cosmology, kwmap=0.02, wiggle=True)
    >>> s = _sigma_squared(m, k, Pk, cosmology, 0)
    >>> fst = mass.sheth_tormen_collapse_function(s, params=(0.5, 1, 0, 1.686))
    >>> mass.halo_mass_function(fst, m_min, m_max, 1.686, 0, k, Pk, cosmology,
    ...                    resolution=4)
    array([1.21845778e-23, 6.09429465e-25, 2.80742696e-26, 7.02778618e-29])

    References
    ----------
    .. [1] Mo, H. and van den Bosch, F. and White, S. (2010), Cambridge
        University Press, ISBN: 9780521857932.
    '''
    f_c = collapse_function
    k = wavenumber
    Pk = power_spectrum

    # Log nu as a function of log mass for a given redshift
    m = np.logspace(np.log10(np.min(m_min)), np.log10(np.max(m_max)),
                    num=resolution)

    def log_nu(log_M):
        sigma2 = _sigma_squared(np.exp(log_M), k, Pk, cosmology, redshift)
        return np.log(delta_c**2 / sigma2) / 2.0

    dlognu_dlogm = np.absolute(derivative(log_nu, np.log(m), dx=step))
    rho_bar = cosmology.critical_density0 * (cosmology.efunc(redshift)) ** 2
    rho_bar = (rho_bar.to(u.kg / u.m ** 3)).value

    return rho_bar * f_c * dlognu_dlogm / np.power(m, 2)


def halo_mass_sampler(mass_function, m_min, m_max, resolution=100, size=None):
    r'''Halo mass sampler.
    This function samples haloes from their mass function,
    see equation 7.46 in [1]_.

    Parameters
    -----------
    mass_function: (nm,) array_like
        Array storing the values of the halo mass function.
    m_min, m_max : float
        Lower and upper bounds for the random variable `m` in solar mass.
    resolution : int, optional
        Resolution of the inverse transform sampling spline. Default is 100.
    size: int, optional
        Output shape of samples. Default is None.

    Returns
    --------
    sample: (nm,) array_like
        Samples drawn from the mass function.

    Examples
    ---------
    >>> import numpy as np
    >>> from skypy.halo import mass
    >>> from skypy.power_spectrum import _eisenstein_hu as eh

    This example will sample from the Sheth and Tormen mass function for a
    Planck15 cosmology

    >>> from astropy.cosmology import Planck15
    >>> cosmology = Planck15
    >>> m_min, m_max = 0.1, 10.0
    >>> m = np.logspace(np.log10(m_min), np.log10(m_max), num=4)
    >>> k = np.logspace(-3, 1, num=5, base=10.0)
    >>> A_s, n_s = 2.1982e-09, 0.969453
    >>> Pk = eh.eisenstein_hu(k, A_s, n_s, cosmology, kwmap=0.02, wiggle=True)
    >>> s = _sigma_squared(m, k, Pk, cosmology, 0)
    >>> fst = mass.sheth_tormen_collapse_function(s, params=(0.5, 1, 0, 1.686))
    >>> mf = mass.halo_mass_function(fst, m_min, m_max, 1.686, 0, k, Pk,
    ...                    cosmology, resolution=4)
    >>> mass.halo_mass_sampler(mf, m_min, m_max, resolution=4)
    0.3595307319328127

    References
    ----------
    .. [1] Mo, H. and van den Bosch, F. and White, S. (2010), Cambridge
        University Press, ISBN: 9780521857932.
    '''
    m = np.logspace(np.log10(np.min(m_min)), np.log10(np.max(m_max)),
                    num=resolution)

    # Sampling from the halo mass function
    PDF = mass_function
    CDF = integrate.cumtrapz(PDF, m, initial=0)
    cdf = CDF/CDF[-1]
    t_lower = np.interp(m_min, m, cdf)
    t_upper = np.interp(m_max, m, cdf)
    n_uniform = np.random.uniform(t_lower, t_upper, size=size)

    return np.interp(n_uniform, cdf, m)


def sheth_tormen_collapse_function(sigma, params):
    r'''Sheth & Tormen collapse fraction.
    This function computes the Sheth & Tormen mass fumction for ellipsoidal
    collapse, see equation 10 in [1]_ or [2]_.

    Parameters
    -----------
    sigma: (ns,) array_like
        Array of the mass variance at different scales and at a given redshift.
    params: float
        The :math:`{A,a,p, delta_c}` parameters of the Sheth-Tormen formalism.

    Returns
    --------
    f_sp: array_like
        Array with the values of the collapse function.

    Examples
    ---------
    >>> import numpy as np

    References
    ----------
    .. [1] R. K. Sheth and G. Tormen,  Mon. Not. Roy. Astron. Soc. 308, 119
        (1999), astro-ph/9901122.
    .. [2] https://www.slac.stanford.edu/econf/C070730/talks/
        Wechsler_080207.pdf
    '''
    A, a, p, delta_c = params
    x = np.power(delta_c / sigma, 2)

    return (A / (np.sqrt(np.pi) * x)) * (1.0 + 1.0 / (a * x)**p) *\
        np.sqrt(a * x / 2.0) * np.exp(-(a * x / 2.0))


press_schechter_collapse_function = partial(sheth_tormen_collapse_function,
                                            params=(0.5, 1, 0, 1.69))


def press_schechter(n, m_star, size=None, x_min=0.00305,
                    x_max=1100.0, resolution=100):
    """Sampling from Press-Schechter mass function (1974).
    Masses following the Press-Schechter mass function following the
    Press and Schechter [1]_ formalism.

    Parameters
    ----------
    n : float
        The n parameter in the Press-Schechter mass function.
    m_star : float
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
