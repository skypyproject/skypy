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
from functools import partial
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


def halo_mass_function(M, sigma, collapse_function, cosmology):
    r'''Halo mass function.
    This function computes the halo mass function, defined
    in equation 7.46 in [1]_.

    Parameters
    -----------
    M : (nm,)
        Array for the halo mass, in units of solar masses.
    sigma: (ns,) array_like
        Array of the mass variance at different scales and at a given redshift.
    collapse_function: (nm,) array_like
        Collapse function to choose from a variety of models:
        `sheth_tormen_collapse_function`, `press_schechter_collapse_function`.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.

    Returns
    --------
    mass_function: (nm,) array_like
        Halo mass function for a given mass array, cosmology and redshift, in
        units of :math:`Mpc^{-3} M_{Sun}^{-1}`.

    Examples
    ---------
    >>> import numpy as np
    >>> from skypy.halo import mass
    >>> from skypy.power_spectrum import _eisenstein_hu as eh

    This example will compute the halo mass function for spherical collapse
    and a Planck15 cosmology. The power spectrum is given by the Eisenstein
    and Hu fitting formula:

    >>> from astropy.cosmology import Planck15
    >>> cosmology = Planck15
    >>> k = np.logspace(-3, 1, num=5, base=10.0)
    >>> A_s, n_s = 2.1982e-09, 0.969453
    >>> Pk = eh.eisenstein_hu(k, A_s, n_s, cosmology, kwmap=0.02, wiggle=True)

    And the Press-Schechter mass function is computed

    >>> m = 10**np.arange(9.0, 12.0, 2)
    >>> sigma = np.sqrt(_sigma_squared(m, k, Pk, 0, cosmology))
    >>> fps = mass.press_schechter_collapse_function(sigma)
    >>> mass.halo_mass_function(m, sigma, fps, cosmology)
    array([1.29448167e-11, 1.91323946e-13])

    References
    ----------
    .. [1] Mo, H. and van den Bosch, F. and White, S. (2010), Cambridge
        University Press, ISBN: 9780521857932.
    '''
    f_c = collapse_function

    dlognu_dlogm = _dlns_dlnM(sigma, M)
    rho_bar = (cosmology.critical_density(0).to(u.Msun / u.Mpc ** 3)).value
    rho_m0 = cosmology.Om(0) * rho_bar

    return rho_m0 * f_c * dlognu_dlogm / np.power(M, 2)


def halo_mass_sampler(mass_function, M, size=None):
    r'''Halo mass sampler.
    This function samples haloes from their mass function,
    see equation 7.46 in [1]_.

    Parameters
    -----------
    mass_function: (nm,) array_like
        Array storing the values of the halo mass function, in
        units of :math:`Mpc^{-3} M_{Sun}^{-1}`.
    M : (nm,)
        Array for the halo mass, in units of solar masses.
    size: int, optional
        Output shape of samples. Default is None.

    Returns
    --------
    sample: (size,) array_like
        Samples drawn from the mass function, in units of solar masses.

    Examples
    ---------
    >>> import numpy as np
    >>> from skypy.halo import mass
    >>> from skypy.power_spectrum import _eisenstein_hu as eh

    This example will sample from the Press-Schechter halo mass function for
    a Planck15 cosmology. The power spectrum is given by the Eisenstein
    and Hu fitting formula:

    >>> from astropy.cosmology import Planck15
    >>> cosmology = Planck15
    >>> k = np.logspace(-3, 1, num=5, base=10.0)
    >>> A_s, n_s = 2.1982e-09, 0.969453
    >>> Pk = eh.eisenstein_hu(k, A_s, n_s, cosmology, kwmap=0.02, wiggle=True)

    And the Press-Schechter mass function is computed

    >>> m = 10**np.arange(9.0, 12.0, 2)
    >>> sigma = np.sqrt(_sigma_squared(m, k, Pk, 0, cosmology))
    >>> fps = mass.press_schechter_collapse_function(sigma)
    >>> mf = mass.halo_mass_function(m, sigma, fps, cosmology)

    And we draw one sample:

    >>> mass.halo_mass_sampler(mf, m)
    56178828376.46093

    References
    ----------
    .. [1] Mo, H. and van den Bosch, F. and White, S. (2010), Cambridge
        University Press, ISBN: 9780521857932.
    '''
    m_min, m_max = np.min(M), np.max(M)

    # Sampling from the halo mass function
    PDF = mass_function
    CDF = integrate.cumtrapz(PDF, M, initial=0)
    cdf = CDF/CDF[-1]
    t_lower = np.interp(m_min, M, cdf)
    t_upper = np.interp(m_max, M, cdf)
    n_uniform = np.random.uniform(t_lower, t_upper, size=size)

    return np.interp(n_uniform, cdf, M)


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
    >>> from skypy.halo import mass
    >>> from skypy.power_spectrum import _eisenstein_hu as eh

    This example will compute the Press-Schecter function for
    spherical collapse and a Planck15 cosmology. The power spectrum is
    given by the Eisenstein and Hu fitting formula:

    >>> from astropy.cosmology import Planck15
    >>> cosmology = Planck15
    >>> k = np.logspace(-3, 1, num=5, base=10.0)
    >>> A_s, n_s = 2.1982e-09, 0.969453
    >>> Pk = eh.eisenstein_hu(k, A_s, n_s, cosmology, kwmap=0.02, wiggle=True)

    And the Press-Schechter function is computed

    >>> m = 10**np.arange(9.0, 12.0, 2)
    >>> sigma = np.sqrt(_sigma_squared(m, k, Pk, 0, cosmology))
    >>> mass.press_schechter_collapse_function(sigma)
    array([0.17896132, 0.21613726])

    References
    ----------
    .. [1] R. K. Sheth and G. Tormen,  Mon. Not. Roy. Astron. Soc. 308, 119
        (1999), astro-ph/9901122.
    .. [2] https://www.slac.stanford.edu/econf/C070730/talks/
        Wechsler_080207.pdf
    '''
    A, a, p, delta_c = params

    return A * np.sqrt(2.0 * a / np.pi) * (delta_c / sigma) * \
        np.exp(- 0.5 * a * (delta_c / sigma)**2) * \
        (1.0 + np.power(np.power(sigma / delta_c, 2.0) / a, p))


press_schechter_collapse_function = partial(sheth_tormen_collapse_function,
                                            params=(0.5, 1, 0, 1.69))


def _sigma_squared(M, wavenumber, linear_power_today, redshift, cosmology):
    k = wavenumber
    Pk = linear_power_today

    if isinstance(M, np.ndarray):
        M = M[:, np.newaxis]

    # The linear frowth function
    D0 = growth_function(0, cosmology)
    Dz = growth_function(redshift, cosmology) / D0

    # Matter mean density today
    rho_bar = (cosmology.critical_density(0).to(u.Msun / u.Mpc ** 3)).value
    rho_m0 = cosmology.Om(0) * rho_bar

    R = np.power(3 * M / (4 * np.pi * rho_m0), 1.0 / 3.0)
    top_hat = 3. * (np.sin(k * R) - k * R * np.cos(k * R)) / ((k * R)**3.)
    integrand = Pk * np.power(top_hat * k, 2)

    return Dz**2 * integrate.simps(integrand, k) / (2. * np.pi**2.)


def _dlns_dlnM(sigma, M):
    ds = np.gradient(sigma, M)
    return np.absolute((M / sigma) * ds)


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
