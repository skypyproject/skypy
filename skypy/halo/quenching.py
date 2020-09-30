"""Galaxy quenching.

This module implements models for environment and mass
quenching by dark matter halos.

Models
======
.. autosummary::
   :nosignatures:
   :toctree: ../api/

   environment_quenched
   mass_quenched
"""

import numpy as np
from scipy import special

__all__ = [
    'environment_quenched',
    'mass_quenched',
    ]


def environment_quenched(nh, probability):
    r'''Environment quenching function.
    This function implements the model proposed by A.Amara where the
    probability of a subhalo being quenched is a fixed
    probability. The model is inspired on [1]_ and [2]_.

    Parameters
    ----------
    nh: integer
        Number of subhalos.
    probability: float
        Quenching probability.

    Returns
    -------
    quenched: (nh,) array_like,  boolean
        Boolean array indicating which subhalo's host galaxies are
        (satellite) environment-quenched.

    Examples
    ---------

    This example shows how many subhalos are environtment quenched (True)
    and how many survive (False) from a list of 1000 halos:

    >>> import numpy as np
    >>> from skypy.halo.quenching import environment_quenched
    >>> from collections import Counter
    >>> quenched = environment_quenched(1000, 0.5)
    >>> Counter(quenched)
    Counter({...})

    References
    ----------
    .. [1] Peng et al. 2010, doi 10.1088/0004-637X/721/1/193.
    .. [2] Birrer et al. 2014, arXiv 1401.3162.

    '''

    return np.random.uniform(size=nh) < probability


def mass_quenched(halo_mass, offset, width):
    r'''Mass quenching function.
    This function implements the model proposed by A.Amara where the
    probability of a halo being quenched is related to the error function
    of the logarithm of the halo's mass standardised by an offset and width
    parameter.  The model is inspired on [1]_ and [2]_.

    Parameters
    ----------
    halo_mass: (nh,) array_like
        Array of halo masses in units of solar mass, Msun.
    offset: float
        Halo mass in Msun at which quenching probability is 50%.
    width: float
        Width of the error function.

    Returns
    -------
    quenched: (nh,) array_like, boolean
        Boolean array indicating which halo's host galaxies are mass-quenched.

    Examples
    ---------

    This example shows how many halos are mass quenched (True)
    and how many survive (False) from a list of 1000 halos:

    >>> import numpy as np
    >>> from astropy import units
    >>> from skypy.halo.quenching import mass_quenched
    >>> from collections import Counter
    >>> offset, width = 1.0e12, 0.5
    >>> halo_mass = np.random.lognormal(mean=np.log(offset), sigma=width,
    ...                                 size=1000)
    >>> quenched = mass_quenched(halo_mass, offset, width)
    >>> Counter(quenched)
    Counter({...})

    References
    ----------
    .. [1] Peng et al. 2010, doi 10.1088/0004-637X/721/1/193.
    .. [2] Birrer et al. 2014, arXiv 1401.3162.

    '''

    standardised_mass = np.log10(halo_mass / offset) / width
    probability = 0.5 * (1.0 + special.erf(standardised_mass/np.sqrt(2)))
    nh = len(halo_mass)
    return np.random.uniform(size=nh) < probability
