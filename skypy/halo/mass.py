r'''Halo mass sampler.
This code samples halos from their mass function.

Models
======
.. autosummary::
   :nosignatures:
   :toctree: ../api/

   ellipsoidal_collapse_function
   halo_mass_function
   halo_mass_sampler
   number_subhalos
   press_schechter
   press_schechter_collapse_function
   press_schechter_mass_function
   sheth_tormen
   sheth_tormen_collapse_function
   sheth_tormen_mass_function
   subhalo_mass_sampler
'''

import numpy as np
from scipy import integrate
from scipy.special import gamma
from skypy.utils.special import gammaincc
from functools import partial
from astropy import units
from skypy.utils.random import schechter

__all__ = [
    'ellipsoidal_collapse_function',
    'halo_mass_function',
    'halo_mass_sampler',
    'number_subhalos',
    'press_schechter',
    'press_schechter_collapse_function',
    'press_schechter_mass_function',
    'sheth_tormen',
    'sheth_tormen_collapse_function',
    'sheth_tormen_mass_function',
    'subhalo_mass_sampler',
 ]


def halo_mass_function(M, wavenumber, power_spectrum, growth_function,
                       cosmology, collapse_function, params):
    r'''Halo mass function.
    This function computes the halo mass function, defined
    in equation 7.46 in [1]_.

    Parameters
    -----------
    M : (nm,) array_like
        Array for the halo mass, in units of solar mass.
    wavenumber : (nk,) array_like
        Array of wavenumbers at which the power spectrum is evaluated,
        in units of Mpc-1.
    power_spectrum: (nk,) array_like
        Linear power spectrum at redshift 0 in Mpc3.
    growth_function : float
        The growth function evaluated at a given redshift for the given
        cosmology.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.
    collapse_function: function
        Collapse function to choose from a variety of models:
        `sheth_tormen_collapse_function`, `press_schechter_collapse_function`.
    params: tuple
        List of parameters that determines the model used for
        the collapse function.

    Returns
    --------
    mass_function: (nm,) array_like
        Halo mass function for a given mass array, cosmology and redshift, in
        units of Mpc-3 Msun-1.

    Examples
    ---------
    >>> import numpy as np
    >>> from skypy.halo import mass
    >>> from skypy.power_spectrum import eisenstein_hu

    This example will compute the halo mass function for elliptical and
    spherical collapse, for a Planck15 cosmology at redshift 0.
    The power spectrum is given by the Eisenstein and Hu fitting formula:

    >>> from astropy.cosmology import Planck15
    >>> cosmo = Planck15
    >>> D0 = 1.0
    >>> k = np.logspace(-3, 1, num=1000, base=10.0)
    >>> A_s, n_s = 2.1982e-09, 0.969453
    >>> Pk = eisenstein_hu(k, A_s, n_s, cosmo, kwmap=0.02, wiggle=True)

    The Sheth and Tormen mass function at redshift 0:

    >>> m = 10**np.arange(9.0, 12.0, 2)
    >>> mass.sheth_tormen_mass_function(m, k, Pk, D0, cosmo)
    array([3.07523240e-11, 6.11387743e-13])

    And the Press-Schechter mass function at redshift 0:

    >>> mass.press_schechter_mass_function(m, k, Pk, D0, cosmo)
    array([3.46908809e-11, 8.09874945e-13])

    For any other collapse models:

    >>> params_model = (0.3, 0.7, 0.3, 1.686)
    >>> mass.halo_mass_function(m, k, Pk, D0, cosmo,
    ...     mass.ellipsoidal_collapse_function, params=params_model)
    array([2.85598921e-11, 5.67987501e-13])

    References
    ----------
    .. [1] Mo, H. and van den Bosch, F. and White, S. (2010), Cambridge
        University Press, ISBN: 9780521857932.
    '''
    sigma = np.sqrt(_sigma_squared(M, wavenumber, power_spectrum,
                                   growth_function, cosmology))
    f_c = collapse_function(sigma, params)

    dlognu_dlogm = _dlns_dlnM(sigma, M)
    rho_bar = (cosmology.critical_density0.to(units.Msun / units.Mpc ** 3)).value
    rho_m0 = cosmology.Om0 * rho_bar

    return rho_m0 * f_c * dlognu_dlogm / np.power(M, 2)


def halo_mass_sampler(m_min, m_max, resolution, wavenumber, power_spectrum,
                      growth_function, cosmology,
                      collapse_function, params, size=None):
    r'''Halo mass sampler.
    This function samples haloes from their mass function,
    see equation 7.46 in [1]_.

    Parameters
    -----------
    m_min, m_max : array_like
        Lower and upper bounds for the random variable m.
    resolution: int
        Resolution of the inverse transform sampling spline.
    wavenumber : (nk,) array_like
        Array of wavenumbers at which the power spectrum is evaluated,
        in units of Mpc-1.
    power_spectrum: (nk,) array_like
        Linear power spectrum at redshift 0 in Mpc3.
    growth_function : float
        The growth function evaluated at a given redshift for the given
        cosmology.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.
    collapse_function: function
        Collapse function to choose from a variety of models:
        `sheth_tormen_collapse_function`, `press_schechter_collapse_function`.
    params: tuple
        List of parameters that determines the model used for
        the collapse function.
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
    >>> from skypy.power_spectrum import eisenstein_hu

    This example will sample from the halo mass function for
    a Planck15 cosmology at redshift 0. The power spectrum is given
    by the Eisenstein and Hu fitting formula:

    >>> from astropy.cosmology import Planck15
    >>> cosmo = Planck15
    >>> D0 = 1.0
    >>> k = np.logspace(-3, 1, num=100, base=10.0)
    >>> A_s, n_s = 2.1982e-09, 0.969453
    >>> Pk = eisenstein_hu(k, A_s, n_s, cosmo, kwmap=0.02, wiggle=True)

    Sampling from the Sheth and Tormen mass function:

    >>> halo_mass = mass.sheth_tormen(1e9, 1e12, 100, k, Pk, D0, cosmo)

    And from the Press-Schechter mass function:

    >>> halo_mass = mass.press_schechter(1e9, 1e12, 100, k, Pk, D0, cosmo)

    For any other collapse models:

    >>> params_model = (0.3, 0.7, 0.3, 1.686)
    >>> halo_mass = mass.halo_mass_sampler(1e9, 1e12, 100, k, Pk, D0, cosmo,
    ...     mass.ellipsoidal_collapse_function, params=params_model)

    References
    ----------
    .. [1] Mo, H. and van den Bosch, F. and White, S. (2010), Cambridge
        University Press, ISBN: 9780521857932.
    '''
    m = np.logspace(np.log10(m_min), np.log10(m_max), resolution)

    massf = halo_mass_function(
            m, wavenumber, power_spectrum, growth_function,
            cosmology, collapse_function, params=params)

    CDF = integrate.cumtrapz(massf, m, initial=0)
    CDF = CDF / CDF[-1]
    n_uniform = np.random.uniform(size=size)
    return np.interp(n_uniform, CDF, m)


def ellipsoidal_collapse_function(sigma, params):
    r'''Ellipsoidal collapse function.
    This function computes the mass function for ellipsoidal
    collapse, see equation 10 in [1]_ or [2]_.

    Parameters
    -----------
    sigma: (ns,) array_like
        Array of the mass variance at different scales and at a given redshift.
    params: float
        The :math:`{A,a,p, delta_c}` parameters of the Sheth-Tormen formalism.

    Returns
    --------
    f_c: array_like
        Array with the values of the collapse function.

    Examples
    ---------
    >>> import numpy as np
    >>> from skypy.halo import mass
    >>> from skypy.power_spectrum import eisenstein_hu
    >>> from skypy.power_spectrum import growth_function

    This example will compute the mass function for
    ellipsoidal collapse and a Planck15 cosmology at redshift 0.
    The power spectrum is given by the Eisenstein and Hu fitting formula:

    >>> from astropy.cosmology import Planck15
    >>> cosmo = Planck15
    >>> D0 = 1.0
    >>> k = np.logspace(-3, 1, num=5, base=10.0)
    >>> A_s, n_s = 2.1982e-09, 0.969453
    >>> Pk = eisenstein_hu(k, A_s, n_s, cosmo, kwmap=0.02, wiggle=True)

    The Sheth-Tormen collapse function at redshift 0:

    >>> m = 10**np.arange(9.0, 12.0, 2)
    >>> sigma = np.sqrt(_sigma_squared(m, k, Pk, D0, cosmo))
    >>> mass.sheth_tormen_collapse_function(sigma)
    array([0.17947815, 0.19952375])

    And the Press-Schechter collapse function at redshift 0:

    >>> mass.press_schechter_collapse_function(sigma)
    array([0.17896132, 0.21613726])

    For any other collapse models:

    >>> params_model = (0.3, 0.7, 0.3, 1.686)
    >>> mass.ellipsoidal_collapse_function(sigma, params=params_model)
    array([0.16667541, 0.18529452])

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


press_schechter_collapse_function = partial(ellipsoidal_collapse_function,
                                            params=(0.5, 1, 0, 1.69))
sheth_tormen_collapse_function = partial(ellipsoidal_collapse_function,
                                         params=(0.3222, 0.707, 0.3, 1.686))
sheth_tormen_mass_function = partial(
                             halo_mass_function,
                             collapse_function=ellipsoidal_collapse_function,
                             params=(0.3222, 0.707, 0.3, 1.686))
press_schechter_mass_function = partial(
                                halo_mass_function,
                                collapse_function=ellipsoidal_collapse_function,
                                params=(0.5, 1, 0, 1.69))
sheth_tormen = partial(halo_mass_sampler,
                       collapse_function=ellipsoidal_collapse_function,
                       params=(0.3222, 0.707, 0.3, 1.686))
press_schechter = partial(halo_mass_sampler,
                          collapse_function=ellipsoidal_collapse_function,
                          params=(0.5, 1, 0, 1.69))


def _sigma_squared(M, k, Pk, growth_function, cosmology):
    M = np.atleast_1d(M)[:, np.newaxis]

    # Growth function
    Dz2 = np.power(growth_function, 2)

    # Matter mean density today
    rho_bar = (cosmology.critical_density0.to(units.Msun / units.Mpc ** 3)).value
    rho_m0 = cosmology.Om0 * rho_bar

    R = np.power(3 * M / (4 * np.pi * rho_m0), 1.0 / 3.0)
    top_hat = 3. * (np.sin(k * R) - k * R * np.cos(k * R)) / ((k * R)**3.)
    integrand = Pk * np.power(top_hat * k, 2)

    return Dz2 * integrate.simps(integrand, k) / (2. * np.pi**2.)


def _dlns_dlnM(sigma, M):
    ds = np.gradient(sigma, M)
    return np.absolute((M / sigma) * ds)


def number_subhalos(halo_mass, alpha, beta, gamma_M, x, m_min, noise=True):
    r'''Number of subhalos.
    This function calculates the number of subhalos above a given initial mass
    for a parent halo of given mass according to the model of Vale & Ostriker
    2004 [1]_ equation (7). The number of subhalos returned can optionally be
    sampled from a Poisson distribution with that mean.

    Parameters
    -----------
    halo_mass : (nm, ) array_like
        The mass of the halo parent, in units of solar mass.
    alpha, beta : float
        Parameters that determines the subhalo Schechter function. Its the amplitude
        is defined by equation (2) in [1].
    gamma_M : float
        Present day mass fraction in subhalos.
    x : float
        Parameter that accounts for the added mass of the original, unstripped
        subhalos.
    m_min : array_like
        Original mass of the least massive subhalo, in units of solar mass.
        Current stripped mass is given by :math:`x m_{min}`.
    noise : bool, optional
        Poisson-sample the number of subhalos. Default is `True`.

    Returns
    --------
    nsubhalos: array_like
        Array of the number of subhalos assigned to parent halos with mass halo_mass.

    Examples
    ---------
    >>> import numpy as np
    >>> from skypy.halo import mass

    This gives the number of subhalos in a parent halo of mass :math:`10^{12} M_\odot`

    >>> halo, min_sh = 1.0e12, 1.0e6
    >>> alpha, beta, gamma_M = 1.9, 1.0, 0.3
    >>> x = 3
    >>> nsh = mass.number_subhalos(halo, alpha, beta, gamma_M, x, min_sh)

    References
    ----------
    .. [1] Vale, A. and Ostriker, J.P. (2005), arXiv: astro-ph/0402500.
    '''
    # m_min is the minimum original subhalo mass
    x_low = m_min / (x * beta * halo_mass)
    # Subhalo amplitude from equation [2]
    A = gamma_M / (beta * gamma(2.0 - alpha))
    # The mean number of subhalos above a mass threshold
    # can be obtained by integrating equation (1) in [1]
    n_subhalos = A * gammaincc(1.0 - alpha, x_low) * gamma(1.0 - alpha)

    # Random number of subhalos following a Poisson distribution
    # with mean n_subhalos
    return np.random.poisson(n_subhalos) if noise else n_subhalos


def subhalo_mass_sampler(halo_mass, nsubhalos, alpha, beta,
                         x, m_min, resolution=100):
    r'''Subhalo mass sampler.
    This function samples the original, unstriped masses of subhaloes from the
    subhalo mass function of their parent halo with a constant mass stripping
    factor given by equation (1) in [1]_ and the upper subhalo mass limit from
    [2]_.

    Parameters
    -----------
    halo_mass : (nm, ) array_like
        The mass of the halo parent, in units of solar mass.
    nsubhalos: (nm, ) array_like
        Array of the number of subhalos assigned to parent halos with mass `halo_mass`.
    alpha, beta : float
        Parameters that determines the subhalo Schechter function. Its the amplitude
        is defined by equation 2 in [1].
    x : float
        Parameter that accounts for the added mass of the original, unstripped
        subhalos.
    m_min : float
        Original mass of the least massive subhalo, in units of solar mass.
        Current stripped mass is given by :math:`m_{min} / x`.
    resolution: int, optional
        Resolution of the inverse transform sampling spline. Default is 100.

    Returns
    --------
    sample: (nh, ) array_like
        List of original masses drawn from the subhalo mass function for each
        parent halo, in units of solar mass. The length corresponds to the total
        number of subhalos for all parent halos, i.e. `np.sum(nsubhalos)`.

    Examples
    ---------
    >>> import numpy as np
    >>> from skypy.halo import mass

    This example samples 100 subhalos for a parent halo of mass 1.0E12 Msun:

    >>> halo, min_sh = 1.0e12, 1.0e6
    >>> alpha, beta, gamma_M = 1.9, 1.0, 0.3
    >>> x = 3
    >>> nsh = 100
    >>> sh = mass.subhalo_mass_sampler(halo, nsh, alpha, beta, x, min_sh)

    References
    ----------
    .. [1] Vale, A. and Ostriker, J.P. (2004), arXiv: astro-ph/0402500.
    .. [2] Vale, A. and Ostriker, J.P. (2004), arXiv: astro-ph/0511816.
    '''
    halo_mass = np.atleast_1d(halo_mass)
    nsubhalos = np.atleast_1d(nsubhalos)
    subhalo_list = []
    for M, n in zip(halo_mass, nsubhalos):
        x_min = m_min / (x * beta * M)
        x_max = 0.5 / (x * beta)
        subhalo_mass = schechter(alpha, x_min, x_max, resolution, size=n, scale=x * beta * M)
        subhalo_list.append(subhalo_mass)
    return np.concatenate(subhalo_list)
