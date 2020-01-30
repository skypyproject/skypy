import numpy as np


def carroll(redshift, cosmology):
    """
    Return the growth function as a function of redshift for a given cosmology
    as approximated by Carroll, Press & Turner (1992) Equation. 29.

    Parameters
    ----------
    redshift : numpy.ndarray
        Array of redshifts at which to evaluate the growth function.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.

    Returns
    -------
    growth : numpy.ndarray
        Array of values for the growth function evaluated at the input
        redshifts for the given cosmology.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import default_cosmology
    >>> redshift = np.array([0, 1, 2])
    >>> cosmology = default_cosmology.get()
    >>> carroll(redshift, cosmology)
    array([0.78136173, 0.47628062, 0.32754955])

    Reference
    ---------
    doi : 10.1146/annurev.aa.30.090192.002435

    """
    Om = cosmology.Om(redshift)
    Ode = cosmology.Ode(redshift)
    growth = 2.5 * Om / (1 + redshift)
    growth = growth / (np.power(Om, 4.0/7.0) - Ode + (1 + 0.5*Om) * (1.0 + Ode/70.0))
    return growth
