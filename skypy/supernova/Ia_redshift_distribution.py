r"""Creates the redshift distribution of SNe Ia given a volumetric rate
=================
.. autosummary::
   :nosignatures:
   :toctree: ../api/
   SNeIa_redshifts

Models
======
.. autosummary::
   :nosignatures:
   :toctree: ../api/
   Ia_redshift_dist
"""
import numpy as np
from astropy import units
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import random

def Ia_redshift_dist(zmax, cosmology, time=365.25, area=10., rate_function=lambda z: 2.47.e-5):
    """Generates an intrisic redshift distribution and number of tranisents,
    given an input volumetric rate, number of days and sky area.
    Parameters
    ----------
    zmax : float
        Maximum redshift at which to calculate the SNe Ia rate
    cosmology : instance
        Instance of an Astropy Cosmology class.
    time : float, optional
        Time in days (default is 1 year or 365.25 days), observer frame
    area : float, optional
        Area in square degrees (default is 10 square degrees).
    rate_function : callable
        A function that accepts a single float (redshift) and returns the
        comoving volumetric rate at each redshift in units of yr^-1 Mpc^-3.
        Default is to set the rate to 2.47e-5 yr^-1 Mpc^-3 for all redshifts (accurate to z~0.1)

    Returns
    -------
    redshift_dist : numpy.array
    
    Examples
    --------
    >>> 
    >>> 
    >>> 
    """
    
    # Get the comoving volume for each redshift shell
    z_bins = 100
    z_binedges = np.linspace(0., zmax, z_bins + 1)
    z_bincentre = np.array(0.5 * (z_binedges[1:] + z_binedges[:-1]))
    
    vol_sphere = np.array(cosmology.comoving_volume(z_binedges).value).real
    vol_shell = vol_sphere[1:] - vol_sphere[:-1]
    
    # SN / (observer year) in shell
    rate_in_shell = np.array(vol_shell * rate_function(z_bincentre).value / (1.+z_bincentre))

    # SN / (observer year) within z_binedges
    vol_rate = np.zeros_like(z_binedges)
    vol_rate[1:] = np.add.accumulate(rate_in_shell)

    # Create a percent point function.
    snrate_cdf = vol_rate / vol_rate[-1]
    snrate_ppf = InterpolatedUnivariateSpline(snrate_cdf, z_binedges, k=1)

    print(snrate_ppf)

    # Total numbe of SNe
    print(area/(4. * np.pi * (180. / np.pi) ** 2))
    print(vol_rate[-1])
    nsne = vol_rate[-1] * (time/365.25) * (area/(4. * np.pi * (180. / np.pi) ** 2))
    print(nsne)

    #for i in range(random.poisson(nsne)):
    #    yield float(snrate_ppf(random.random()))
    redshifts=np.array(list([float(snrate_ppf(random.random())) for i in range(random.poisson(nsne))]))

    return redshifts