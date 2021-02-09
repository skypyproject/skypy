'''Halo mass sampler.
This code samples cluster halos from the mass function (Despali+ 16).

'''
import numpy as np
import colossus as colossus
import astropy as astropy
from astropy.cosmology import WMAP9
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy import integrate
import scipy


def selection(z):
    """
        eRosita selection function from Pillepich et al. 2012.
        Including mass cut in the low mass end.
        func(x) is a fitting function based on the data point presented in Pillepich et al. 2012.
    Parameters
    -----------
    z: float
     Redshift
            
    Returns
    --------
    fit_M: array_like
     minimum halo mass (mdef: 500c, unit: Msun) that can be detected by eRosita

    """
    
    def func(x):
        a = 9.26958708e+04
        b = -4.55726030e+00
        c = 1.52774801e+00
        d = -1.85377292e+05
        return a* (1+scipy.special.erf(((x)- b ) / c )  ) + d
    
    if z <= 0.1:  ##apply mass cut: remove halo with halo mass M500c<1e+13
        fit_M = 1e+13
    else:
        p = func(z)
        fit_M = 10**(p)
    return fit_M  ##unit: Msun

def mass_sampler(z, size, mdef = '500c', m_min = 1e+12, m_max = 1e+16, select = False, newcosmo_astropy = WMAP9, sigma8 = 0.8200, ns = 0.9608):
    """
        This function generate a sample of cluster with halo mass from Despali+16 mass function.
       
    Parameters
    -----------
    z: float
     Redshift
    size: int
     Number of clusters we need.
    mdef: str
     halo mass definition, any spherical overdensity. For example: '200m', 200c', '500c', '2500c'
         ***If select=True, then mdef has to be '500c'.
    m_min, m_max: float
     Lower and upper bounds for cluster mass (Unit: Msun).
    select: bool
     If true, apply the eRosita selection function to the mass function
    newcosmo_astropy: astropy.cosmology.Cosmology
    sigma8: float
    ns: float
    
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
    >>>mass_sampler(z=0.1,size=100,select=True)
    
    """
    cosmo = colossus.cosmology.cosmology.fromAstropy(newcosmo_astropy, sigma8 = sigma8, ns = ns, name = 'my_cosmo')
    h0 = newcosmo_astropy.h
    m_h0 = np.logspace(np.log10(m_min*h0), np.log10(m_max*h0), 1000) ##unit: Msun/h
    dndm = mass_function.massFunction(m_h0, z, mdef = mdef, model = 'despali16',q_out = 'dndlnM',q_in='M')*h0**3/m_h0*h0  ##unit: 1/Msun/Mpc^3
    m = m_h0/h0
    if select==True:
        dndm_se = dndm*(np.heaviside([m - selection(z)], 1.0))
        dndm_se = np.reshape(dndm_se, (len(m)))
    else:
        dndm_se = dndm
    CDF = integrate.cumtrapz(dndm_se, (m), initial=0)
    CDF = CDF / CDF[-1]
    n_uniform = np.random.uniform(size = size)
    masssample = np.interp(n_uniform, CDF, m)
    return masssample

