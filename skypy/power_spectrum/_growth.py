from astropy.utils import isiterable
import numpy as np
from scipy import integrate


__all__ = [
   'growth_factor',
   'growth_function',
   'growth_function_carroll',
   'growth_function_derivative',
]


def growth_function_carroll(redshift, cosmology):
    """Computation of the growth function.

    Return the growth function as a function of redshift for a given cosmology
    as approximated by Carroll, Press & Turner (1992) Equation 29.

    Parameters
    ----------
    redshift : array_like
        Array of redshifts at which to evaluate the growth function.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.

    Returns
    -------
    growth : numpy.ndarray, or float if input scalar
        The growth function evaluated at the input redshifts for the given
        cosmology.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import default_cosmology
    >>> redshift = np.array([0, 1, 2])
    >>> cosmology = default_cosmology.get()
    >>> growth_function_carroll(redshift, cosmology)
    array([0.781361..., 0.476280..., 0.327549...])

    References
    ----------
    doi : 10.1146/annurev.aa.30.090192.002435
    """
    if isiterable(redshift):
        redshift = np.asarray(redshift)
    if np.any(redshift < 0):
        raise ValueError('Redshifts must be non-negative')

    Om = cosmology.Om(redshift)
    Ode = cosmology.Ode(redshift)
    Dz = 2.5 * Om / (1 + redshift)
    return Dz / (np.power(Om, 4.0/7.0) - Ode + (1 + 0.5*Om) * (1.0 + Ode/70.0))


def growth_factor(redshift, cosmology, gamma=6.0/11.0):
    """Computation of the growth factor.

    Function used to calculate f(z), parametrised growth factor at different
    redshifts, as described in [1]_ equation 17.

    Parameters
    ----------
    redshift : array_like
        Array of redshifts at which to evaluate the growth function.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.
    gamma : float
        Growth index providing an efficient parametrization of the matter
        perturbations.

    Returns
    -------
    growth_factor : ndarray, or float if input scalar
      The redshift scaling of the growth factor.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmology = FlatLambdaCDM(H0=67.04, Om0=0.3183, Ob0=0.047745)
    >>> growth_factor(0, cosmology)
    0.5355746155304598

    References
    ----------
    .. [1] E. V. Linder, Phys. Rev. D 72, 043529 (2005)
    """
    z = redshift

    omega_m_z = cosmology.Om(z)
    growth_factor = np.power(omega_m_z, gamma)

    return growth_factor


def growth_function(redshift, cosmology, gamma=6.0/11.0):
    """Computation of the growth function.

    Function used to calculate D(z), growth function at different redshifts,
    as described in [1]_ equation 16.

    Parameters
    ----------
    redshift : array_like
        Array of redshifts at which to evaluate the growth function.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.
    gamma : float
        Growth index providing an efficient parametrization of the matter
        perturbations.

    Returns
    -------
    growth_function : ndarray
      The redshift scaling of the growth function.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import integrate
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmology = FlatLambdaCDM(H0=67.04, Om0=0.3183, Ob0=0.047745)
    >>> growth_function(0, cosmology)
    0.7909271056297236

    References
    ----------
    .. [1] E. V. Linder, Phys. Rev. D 72, 043529 (2005)
    """
    z = redshift

    def integrand(x):
        integrand = (growth_factor(x, cosmology, gamma) - 1) / (1 + x)
        return integrand

    if isinstance(z, int) or isinstance(z, float):
        integral = integrate.quad(integrand, z, 1100)[0]
        g = np.exp(integral)
        growth_function = g / (1 + z)

    elif isinstance(z, np.ndarray):
        growth_function = np.zeros(np.shape(z))

        for i, aux in enumerate(z):
            integral = integrate.quad(integrand, aux, 1100)[0]
            g = np.exp(integral)
            growth_function[i] = g / (1 + aux)

    else:
        raise ValueError('Redshift must be integer, float or ndarray ')

    return growth_function


def growth_function_derivative(redshift, cosmology, gamma=6.0/11.0):
    """Computation of the first derivative of the growth function.

    Function used to calculate D'(z), derivative of the growth function
    with respect to redshift, described in [1]_ equation 16.

    Parameters
    ----------
    redshift : array_like
        Array of redshifts at which to evaluate the growth function.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.
    gamma : float
        Growth index providing an efficient parametrization of the matter
        perturbations.

    Returns
    -------
    growth_function_derivative : ndarray, or float if input scalar
      The redshift scaling of the derivative of the growth function.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import integrate
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmology = FlatLambdaCDM(H0=67.04, Om0=0.3183, Ob0=0.047745)
    >>> growth_function_derivative(0, cosmology)
    -0.42360048051025856

    References
    ----------
    .. [1] E. V. Linder, Phys. Rev. D 72, 043529 (2005)
    """
    z = redshift

    growth_function_derivative = - growth_function(z, cosmology, gamma) * \
        growth_factor(z, cosmology, gamma) / (1.0 + z)

    return growth_function_derivative
