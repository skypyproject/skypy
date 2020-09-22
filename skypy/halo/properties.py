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
from scipy import integrate
from functools import partial
from astropy import units

__all__ = [
    'halo_circular_velocity',
 ]

def halo_circular_velocity(M, Delta_v, redshift, cosmology):
    """Halo circular velocity.
    This function computes the halo circular velocity, setting it 
    equal to the virial velocity using equation (3) from [1]_.

    Parameters
    ----------
    M : (nm,) array_like
    	Array for the halo mass, in units of solar mass.
    	We assume here that the halo mass is equal to the virial mass.
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
    >>> cosmo = Planck15
    >>> m = 10**np.arange(9.0, 12.0, 2)
    >>> properties.halo_circular_velocity(m,!!!!!)
    >>> mass.sheth_tormen_mass_function(m, k, Pk, D0, cosmo)
    array([!!!, !!!])

    References
    ----------
    .. [1] Maller and Bullock 2004 MNRAS 355 694 DOI:10.1111/j.1365-2966.2004.08349.x
    """

    circular_velocity = 96.6 * (Delta_v * cosmology.omega_matter * cosmology.h **2) * 
    	pow((1 + redshift)/0.3,0.5) * pow(M/1e11,1/3)

    return circular_velocity
