"""Halo properties module.

This module provides methods to add simple properties to halos

Models
======

.. autosummary::
   :nosignatures:
   :toctree: ../api/

   halo_circular_velocity

"""

import numpy as np
from astropy.units import Unit

__all__ = [
    'halo_circular_velocity',
 ]

def halo_circular_velocity(halo_virial_mass, Delta_v, redshift, cosmology):
    """Halo circular velocity.
    This function computes the halo circular velocity, setting it 
    equal to the virial velocity using equation (3) and footnote 2 from [1]_.

    Parameters
    ----------
    halo_virial_mass : (nm,) array_like
    	Array for the virial mass, in units of solar mass.
    Delta_v : (nm,) array_like
        The mean overdensity of the halo.
    redshift : (nm,) array_like
        The redshift of each halo.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.
    
    Returns
    --------
    circular_velocity: (nm,) array_like
        Halo circular velocity for a given mass array, cosmology and redshift, in
        units of km s-1.

    Examples
    ---------
    >>> import numpy as np
    >>> from skypy.halo import properties

    This example will compute the halo circular velocity, for a Planck15 cosmology at redshift 0.

    >>> from astropy.cosmology import Planck15
    >>> cosmology = Planck15
    >>> halo_virial_mass = 10**np.arange(9.0, 12.0, 2) * Unit.Msun
    >>> Delta_v = np.arange(201.0, 201.1, 0.1)
    >>> redshift = np.arange(0.3, 1, 0.5)
    >>> properties.halo_circular_velocity(halo_virial_mass, Delta_v, redshift, cosmology)
    <Quantity [ 6.11303684, 36.72661831] km2 / (Mpc2 s2)>

    References
    ----------
    .. [1] Barnes and Haehnelt 2010 equation 3 https://arxiv.org/pdf/1403.1873.pdf
    """

    virial_velocity = 96.6 * Unit('km s-1') * \
        np.power(Delta_v * cosmology.Om0 * cosmology.h**2 / 24.4, 1.0/6.0) * \
        np.sqrt((1 + redshift) / 3.3) * \
        np.power(halo_virial_mass / (1.0e11 * Unit('Msun')), 1.0/3.0)

    return virial_velocity
