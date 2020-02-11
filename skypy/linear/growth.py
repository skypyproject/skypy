"""Linear growth module.
This module provides facilities to evaluate the cosmological linear growth
function and related quantities.
"""

from astropy.utils import isiterable
import numpy as np


def growth_function_carroll(redshift, cosmology):
    """
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
    array([0.78136173, 0.47628062, 0.32754955])

    Reference
    ---------
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
