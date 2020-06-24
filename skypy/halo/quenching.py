"""Galaxy quenching.

This module facilitates the implementation of environment and mass
quenching phenomena.
"""

import numpy as np
from scipy import special

__all__ = [
    'environment_quenching',
    'mass_quenching',
    ]


def environment_quenching(number_subhalos, probability=0.5):
    r'''Environment quenching.
    This function implements the model proposed by A.Amara where the
    probability of a subhalo being quenched is a fixed
    probability. The model is inspired on [1]_ and [2]_.

    Parameters
    ----------
    number_subhalos: integer
        Number of subhalos hosting blue star-forming galaxies.
    probability: float, optional
        Fixed “killing” probability. Default is 0.5.
    Returns
    -------
    quenched: boolean, (nh,) array_like
        Boolean array indicating which subhalo's host galaxies are
        (satellite) environment-quenched.

    Examples
    ---------

    This example shows how many subhalos are environtment quenched (True)
    and how many survive (False) from a list of 1000 halos:

    >>> import numpy as np
    >>> import random
    >>> import skypy.halo.quenching as q
    >>> from collections import Counter
    >>> random.seed(42)
    >>> quenched = q.environment_quenching(1000)
    >>> Counter(quenched)
    Counter({True: 521, False: 479})

    References
    ----------
    .. [1] Peng et al. 2010, doi 10.1088/0004-637X/721/1/193.
    .. [2] Birrer et al. 2014, arXiv 1401.3162.

    '''

    quenched = np.zeros(number_subhalos, dtype=bool)
    subhalos_probability = np.random.uniform(0, 1, number_subhalos)

    quenched = subhalos_probability < probability

    return quenched


def mass_quenching(halo_mass, offset, width):
    r'''Mass quenching.
    This function implements the model proposed by A.Amara where the
    probability of a halo being quenched is an error function
    of the logarithm of the halo's mass standardised by an offset and width
    parameter.  The model is inspired on [1]_ and [2]_.

    Parameters
    ----------
    halo_mass: (nh,) array_like
        Array of halo masses.
    offset: float
        Offset parameter (halo mass at which quenching probability = 0.5)
    width: float
        Width parameter.

    Returns
    -------
    quenched: (nh,) array_like, boolean
        Boolean array indicating which halo's host galaxies are mass-quenched.

    Examples
    ---------

    This example shows how many halos are mass quenched (True)
    and how many survive (False) from a list of 1000 halos:

    >>> import numpy as np
    >>> import random
    >>> import skypy.halo.quenching as q
    >>> from collections import Counter
    >>> random.seed(42)
    >>> offset = 12
    >>> width = 6
    >>> halo_mass = np.linspace(0,24, num=1000)
    >>> quenched = q.mass_quenching(halo_mass, offset, width)
    >>> Counter(quenched)
    Counter({False: 486, True: 514})

    References
    ----------
    .. [1] Peng et al. 2010, doi 10.1088/0004-637X/721/1/193.
    .. [2] Birrer et al. 2014, arXiv 1401.3162.

    '''

    standardised_mass = (halo_mass - offset) / width
    probability = special.erf(standardised_mass)

    number_halos = len(halo_mass)
    quenched = np.zeros(number_halos, dtype=bool)
    halos_probability = np.random.uniform(-1, 1, number_halos)

    quenched = halos_probability < probability

    return quenched
