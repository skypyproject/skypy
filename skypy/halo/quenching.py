r'''Galaxy quenching.
This code implement the model proposed by A.Amara for environment and mass
quenching.

Models
======
.. autosummary::
   :nosignatures:
   :toctree: ../api/

   environment_quenching
   mass_quenching
'''

import numpy as np


__all__ = [
     'environment_quenching',
     'mass_quenching',
 ]


def environment_quenching(subhalo_mass, probability=0.5):
    """Environment quenching.
    This function implements the model proposed by A.Amara where the
    probability of a subhalo's host galaxy being quenched is a fixed
    probability.

    Parameters
    ----------
    subhalo_mass: (nh,) array_like
        Array of halo masses.
    probability: float
        Fixed “killing” probability. Default is 0.5.
    Returns
    -------
    quenched: boolean, (nh,) array_like
        Boolean array indicating which subhalo's host galaxies are
        (satellite) environment-quenched.

    Examples
    ---------
    >>> import numpy as np
    >>> import skypy.halo.quenching as q

    References
    ----------
    .. [1] Birrer et al. 2018.

    """
    number_subhalos = len(subhalo_mass)
    quenched = np.zeros(number_subhalos, dtype=bool)
    subhalos_probability = np.random.uniform(0, 1, number_subhalos)

    for s in subhalo_mass:
        if subhalos_probability > probability:
            quenched = True

    return quenched


def mass_quenching(halo_mass, offset, width):
    """Mass quenching.
    This function implements the model proposed by A.Amara where the
    probability of a halo's host galaxy being quenched is an error function
    of the logarithm of the halo's mass standardised by an offset and width
    parameter (c.f. standardising a gaussian distribution by it's mean and
    standard deviation).

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
    quenched: boolean, (nh,) array_like
        Boolean array indicating which halo's host galaxies are mass-quenched.

    Examples
    ---------
    >>> import numpy as np
    >>> import skypy.halo.quenching as q

    References
    ----------
    .. [1] Birrer et al. 2018.

    """
    quenched = True
    return quenched
