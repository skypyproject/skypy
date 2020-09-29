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

__all__ = [
    'halo_circular_velocity',
 ]

def halo_circular_velocity(M, Delta_v, redshift, cosmology):
    """Halo circular velocity.
    This function computes the halo circular velocity, setting it 
    equal to the virial velocity using equation (3) and footnote 2 from [1]_.

    Parameters
    ----------
    M : (nm,) array_like
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
    >>> M = 10**np.arange(9.0, 12.0, 2)
    >>> Delta_v = np.arange(1.0, 1.1, 0.1)
    >>> redshift = np.arange(0.3, 1, 0.5)
    >>> properties.halo_circular_velocity(M, Delta_v, redshift, cosmology)
    <Quantity [ 6.11303684, 36.72661831] km2 / (Mpc2 s2)>

    References
    ----------
    .. [1] Maller and Bullock 2004 MNRAS 355 694 DOI:10.1111/j.1365-2966.2004.08349.x
    
	Notes of things to ask / think about:
	* where to get Delta_v from - do we have a module for this already, or compute it in here too?
	* fix h hack / check h in Maller & Bullock is h100
	* should we call it the circular_velocity or the virial_velocity?
	* ditto M - should we call it M_v or pretend virial mass = halo mass throughout?
	* should we output the virial radius here as well or is this already done elsewhere?
    """

    h = cosmology.H0 / 100
    circular_velocity = 96.6 * (Delta_v * cosmology.Om0 * h **2) * pow((1 + redshift)/0.3,0.5) * pow(M/1e11,1/3)

    return circular_velocity
