"""Halo sampler.
This code samples haloes from their mass function."""

import numpy as np
from scipy import integrate

__all__ = [
     'nu',
 ]


def nu(r_min, r_max, k_min, k_max, number):
    """Nu function.
    This function samples haloes from their mass function following Sheth &
    Tormen formalism, see equation 10 in [1]_ or [2]_.

    Parameters
    -----------
    r_min, r_max : float
        Minimum and maximum values for the viral radius in units of [Mpc].
    k_min, k_max : float
        Minimum and maximum values for the viral wavenumber, units [Mpc^-1].
    number : integer
        Number of  radius and wavenumber elements. Default is 1000.

    Returns
    --------
    nu : numpy.ndarray
       nu in equation () in [].

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

    r = np.linspace(r_min, r_max, number)  # virial radius Mpc/h!!!!!
    k = np.linspace(k_min, k_max, number)

    top_hat = 3. * (np.sin(k * r) - k * r * np.cos(k * r) / (k * r) ** 3)

    integrand = 0.5 * k * k * np.power(top_hat * k, 2) / (np.pi**2)
    sigma_squared = integrate.cumtrapz(integrand, k, initial=1)

    return np.power(1.68647, 2) / sigma_squared


# This bit should go to utils
def unnormPDF(nu):  # (r_min, r_max, k_min, k_max, res):
    nu = np.linspace(0.1, 200, 100)
    uPDF = (0.5 / (np.sqrt(np.pi) * nu)) * (1.0 + 1.0 / (0.707 * nu)**0.3) *\
        np.sqrt(0.707 * nu / 2.0) * np.exp(-(0.707 * nu / 2.0))
    return uPDF


# normalised PDF(nu(sigma(M(R))))
def PDF(nu):
    nu = np.linspace(0.1, 200, 100)
    CDF = integrate.cumtrapz(unnormPDF(nu), nu, initial=0)
    norm = CDF[-1]
    return unnormPDF(nu) / norm


# sampler as a function of nu
def sample_HMF(nu, size):
    nu = np.linspace(0.1, 200, 100)
    CDF = integrate.cumtrapz(unnormPDF(nu), nu, initial=0)
    CDF = CDF/CDF[-1]
    nurand = np.random.uniform(0, 1, size)
    sample = np.interp(nurand, CDF, nu)
    return sample
