import numpy as np
from scipy import integrate
''' This module computes the growth function as a function of redshift, D, as
    described in equation 16 in [1].

    Functions
    ---------
        - Omega matter as a function of redshift, Omegam, equation 6.7 in [2].
        - Growth rate as a function of redshift, f, equation 17 in [1].

    Additional
    ----------
        - Derivative of the growth function with respect to redshift as a
          function of redshift, Dprime, is an optional output.

    References
    ----------
        [1] E. V. Linder, Phys. Rev. D 72, 043529 (2005)
        [2] A. R. Liddle, Chichester, UK: Wiley (1998)
    '''


def growth_factor(redshift, cosmology):
    """ Function used to calculate f(z), parametrised growth factor at different
        redshifts, as described in [1].

        Parameters
        ----------
        redshift : array_like
          Input redshifts.

        Returns
        -------
        growth_factor : ndarray, or float if input scalar
          The redshift scaling of the growth factor, equation 17 in [1].

        References
        ----------
            [1] E. V. Linder, Phys. Rev. D 72, 043529 (2005)

        Examples
        --------
        >>> import numpy as np
        >>> from astropy.cosmology import FlatLambdaCDM
        >>> cosmology = FlatLambdaCDM(H0=67.04, Om0=0.3183, Ob0=0.047745)
        >>> growth_factor(0, cosmology)
        0.5355746155304598
        """
    z = redshift

    gamma = 6./11.
    omega_m_z = cosmology.Om(z)
    growth_factor = np.power(omega_m_z, gamma)

    return growth_factor


def growth_function(redshift, cosmology):
    """ Function used to calculate D(z), growth function at different redshifts,
        as described in [1].

        Parameters
        ----------
        redshift : array_like
          Input redshifts.

        Returns
        -------
        growth_function : ndarray
          The redshift scaling of the growth function, equation 16 in [1].

        References
        ----------
            [1] E. V. Linder, Phys. Rev. D 72, 043529 (2005)

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import integrate
        >>> from astropy.cosmology import FlatLambdaCDM
        >>> cosmology = FlatLambdaCDM(H0=67.04, Om0=0.3183, Ob0=0.047745)
        >>> growth_function(0, cosmology)
        0.7909271056297236
        """
    z = redshift

    integrand = lambda x: (growth_factor(x, cosmology) - 1) / (1 + x)

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
        print('Redshift is not an integer, neither a float, nor a ndarray ')

    return growth_function


def growth_function_derivative(redshift, cosmology):
    """ Function used to calculate D'(z), derivative of the growth function
        with respect to redshift, described in [1].

        Parameters
        ----------
        redshift : array_like
          Input redshifts.

        Returns
        -------
        growth_function_derivative : ndarray, or float if input scalar
          The redshift scaling of the derivative of the growth function.
          Analytic expression derived from equation 16 in [1].

        References
        ----------
            [1] E. V. Linder, Phys. Rev. D 72, 043529 (2005)

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import integrate
        >>> from astropy.cosmology import FlatLambdaCDM
        >>> cosmology = FlatLambdaCDM(H0=67.04, Om0=0.3183, Ob0=0.047745)
        >>> growth_function_derivative(0, cosmology)
        -0.42360048051025856
        """
    z = redshift

    growth_function_derivative = - growth_function(z, cosmology) * \
        growth_factor(z, cosmology) / (1.0 + z)

    return growth_function_derivative
