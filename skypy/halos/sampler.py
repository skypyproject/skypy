'''Halo mass sampler.
This code samples cluster halos from colossus mass function .

'''
import numpy as np
import colossus as colossus
import astropy as astropy
from astropy.cosmology import WMAP9
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy import integrate
import scipy



def colossus_mass_sampler(redshift, model, mdef, m_min, m_max, cosmology, sigma8, ns, size=None, resolution=1000):
    """
        This function generate a sample of cluster with halo mass from Despali+16 mass function.
       
    Parameters
    -----------
    redshift: float
     Redshift
    model: string
     Mass function model which is available in colossus
    mdef: str
     halo mass definition, any spherical overdensity. For example: '200m', 200c', '500c', '2500c'
         ***If select=True, then mdef has to be '500c'.
    m_min, m_max: float
     Lower and upper bounds for cluster mass (Unit: Msun).
    cosmology: astropy.cosmology.Cosmology
    sigma8: float
    ns: float
    size: int
     Number of clusters we need.
     
    Returns
    --------
    halo mass: array_like (unit: Msun)
    
    Examples
    ---------
    >>>import numpy as np
    >>>import colossus as colossus
    >>>import astropy as astropy
    >>>from astropy.cosmology import WMAP9
    >>>from colossus.cosmology import cosmology
    >>>from colossus.lss import mass_function
    >>>from scipy import integrate

    >>>sigma8 = 0.8200
    >>>ns = 0.9608
    >>>colossus_mass_sampler(redshift=0.1, model='despali16', mdef= '500c', m_min = 1e+12, m_max = 1e+16, cosmology=WMAP9, sigma8 = 0.8200, ns = 0.9608, size=100)
    
    """
    cosmo = colossus.cosmology.cosmology.fromAstropy(cosmology, sigma8 = sigma8, ns = ns, name = 'my_cosmo')
    h0 = cosmology.h
    m_h0 = np.logspace(np.log10(m_min*h0), np.log10(m_max*h0), resolution) ##unit: Msun/h
    dndm = mass_function.massFunction(m_h0, redshift, mdef = mdef, model = model,q_out = 'dndlnM',q_in='M')/m_h0
    m = m_h0/h0
    CDF = integrate.cumtrapz(dndm, (m), initial=0)
    CDF = CDF / CDF[-1]
    n_uniform = np.random.uniform(size = size)
    masssample = np.interp(n_uniform, CDF, m)
    return masssample
