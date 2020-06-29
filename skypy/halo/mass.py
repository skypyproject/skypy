r'''Halo mass sampler.
This code samples halos from their mass function.

Models
======
.. autosummary::
   :nosignatures:
   :toctree: ../api/

   press_schechter
   halo_mass_function
   sheth_tormen_collapse_function
   press_schechter_collapse_function
'''

import numpy as np
from scipy import integrate
from functools import partial
from astropy.constants import G
from astropy import units as u

from skypy.power_spectrum import growth_function
from skypy.utils.random import schechter

__all__ = [
    'press_schechter',
    'halo_mass_function',
    'sheth_tormen_collapse_function',
    'press_schechter_collapse_function',
 ]


def _sigma_squared(M, wavenumber, linear_power_today, redshift, cosmology):
    k = wavenumber
    Pk = linear_power_today
    Hz = cosmology.H(redshift)

    R = np.power(2 * G.value * M / Hz, 1.0 / 3.0)
    j1 = (np.sin(k * R) - np.cos(k * R) * k * R) / (k * R)
    top_hat = 3 * j1 / (k * R)**2
    integrand = Pk * top_hat**2.0 * k**2.0
    return integrate.simps(integrand, k) / (2. * np.pi ** 2.)


def halo_mass_function(collapse_function, m_min, m_max, m_star, redshift,
                       wavenumber, power_spectrum, cosmology,
                       resolution=100, size=None):
    """Halo mass function.
    This function samples haloes from their mass function following Sheth &
    Tormen formalism, see equation 7.46 in [1]_.

    Parameters
    -----------
    collapse_function: function
        Collapse function to choose from a variety of models:
        `sheth_tormen_collapse_function`, `press_schechter_collapse_function`.
    m_min, m_max : (nm,) array_like
        Lower and upper bounds for the random variable `m`.
    m_star: float
        Factors parameterising the characteristic mass.
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
    size: int, optional
        Output shape of samples. Default is None.

    Returns
    --------
    sample: array_like
        Samples drawn from the Sheth-Tormen function, equation (10) in [1].

    Examples
    ---------
    >>> import numpy as np
    >>> from skypy.halo import mass

    Note here that the variable :math:`nu` that appears in the Sheth and Tormen
    collapse function is the squared version of the variable :math:`nu` in
    equation 7.46 in [1].

    References
    ----------
    .. [1] Mo, H. and van den Bosch, F. and White, S. (2010), Cambridge
        University Press, ISBN: 9780521857932.
    """
    f_c = collapse_function
    k = wavenumber
    Pk = power_spectrum

    # The linear frowth function
    D0 = growth_function(0, cosmology)
    Dz = growth_function(redshift, cosmology) / D0

    # nu as a function of mass for a given redshift
    m = np.logspace(np.log10(np.min(m_min)), np.log10(np.max(m_max)),
                    resolution)
    delta_critical_squared = _sigma_squared(m_star)
    nu = np.sqrt(delta_critical_squared / _sigma_squared(m)) / Dz

    # Rest of prefactors
    rho_bar = cosmology.critical_density0 * (cosmology.efunc(redshift)) ** 2
    rho_bar = (rho_bar.to(u.kg / u.m ** 3)).value
    dlognu_dlogm = nu * Pk * k  # work on this. This is nothing now

    # Sampling from the halo mass function
    PDF = rho_bar * f_c * dlognu_dlogm / np.power(m, 2)
    CDF = integrate.cumtrapz(PDF, m, initial=0)
    cdf = CDF/CDF[-1]
    t_lower = np.interp(np.min(m), m, cdf)
    t_upper = np.interp(np.max(m), m, cdf)
    n_uniform = np.random.uniform(t_lower, t_upper, size=size)
    n_sample = np.interp(n_uniform, cdf, m)

    return n_sample


def sheth_tormen_collapse_function(sigma, delta_critical, redshift, params):
    r'''Sheth & Tormen collapse fraction.
    This function computes the Sheth & Tormen mass fumction for ellipsoidal
    collapse, see equation 10 in [1]_ or [2]_.

    Parameters
    -----------
    sigma: (ns,) array_like
        Array of the mass variance at different scales and at a given redshift.
    delta_critical: float
        Critical density. Collapsed objects denser than :math:`delta_c` would
        form virialised objects.
    redshift : float
        Redshift value at which to evaluate the mass variance.
    params: float
        The :math:`{A,a,p}` parameters of the Sheth-Tormen formalism.

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
    A, a, p = params
    x = np.power(delta_critical / sigma, 2)

    return (A / (np.sqrt(np.pi) * x)) * (1.0 + 1.0 / (a * x)**p) *\
        np.sqrt(a * x / 2.0) * np.exp(-(a * x / 2.0))


press_schechter_collapse_function = partial(sheth_tormen_collapse_function,
                                            params=(0.5, 1, 0))


def _derivative(f, a, method='central', step=0.01):
    r'''Numerical derivative.
    This function computes the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable.
    a : number
        Compute derivative at :math:`x = a`.
    method : string
        Difference formula: 'forward', 'backward' or 'central'.
    step : number
        Step size in difference formula.

    Returns
    -------
    float

    Notes
    -----
    The difference formula:
        central: :math:`(f(a+h) - f(a-h))/2h`
        forward: :math:`(f(a+h) - f(a))/h`
        backward: :math:`(f(a) - f(a-h))/h`

    Examples
    --------
    >>> import numpy as np
    >>> _derivative(np.cos, 0, method='forward', step=1e-8)
    0.0
    '''
    h = step

    if method == 'central':
        return (f(a + h) - f(a - h)) / (2 * h)
    elif method == 'forward':
        return (f(a + h) - f(a)) / h
    elif method == 'backward':
        return (f(a) - f(a - h)) / h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


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
