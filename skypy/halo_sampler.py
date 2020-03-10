#this code samples haloes from their mass function following Shet&Tormen 1999 formalism (eq.10)
#and https://www.slac.stanford.edu/econf/C070730/talks/Wechsler_080207.pdf

import numpy as np
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

#tophat model
def tophat(r_min, r_max, k_min, k_max, res):
    r = np.linspace(r_min,r_max,res) #virial radius Mpc/h
    k=np.linspace(k_min,k_max,res)
    top = 3.*(np.sin(k*r)-k*r*np.cos(k*r)/(k*r)**3)
    return top

#linear power spectrum
#substitute PowerSpecLin with external function
def PowerSpecLin(k_min, k_max, res):  
    k=np.linspace(k_min,k_max,res)
    return k**2

#sigma(M(R))
def sig(r_min, r_max, k_min, k_max, res):
    r = np.linspace(r_min,r_max,res)
    k = np.linspace(k_min,k_max,res)
    ff = (1./(2.*np.pi**2))*PowerSpecLin(k_min, k_max, res)*(tophat(r_min, r_max, k_min, k_max, res)**2)*(k**2)
    sig2 = integrate.cumtrapz(ff, k, initial=1)
    return np.sqrt(sig2)

#-------------------------- from here on everything is as a function of nu = (1.68647/sigma(M(R)))**2
def nu(r_min, r_max, k_min, k_max, res):
    rr = (1.68647/sig(r_min, r_max, k_min, k_max, res))**2 
    return rr

#unnormalised PDF(nu(sigma(M(R))))
def unnormPDF(nu):#(r_min, r_max, k_min, k_max, res): 
    #nu = (1.68647/sig(r_min, r_max, k_min, k_max, res))**2 
    #uPDF = (1./(2.*nu))*(1.+1./(0.707*nu)**0.3)*np.sqrt(0.707*nu/2.)*np.exp(-(0.707*nu/2.))/np.sqrt(np.pi)
    nu=np.linspace(0.1,200,100)
    uPDF = (1./(2.*nu))*(1.+1./(0.707*nu)**0.3)*np.sqrt(0.707*nu/2.)*np.exp(-(0.707*nu/2.))/np.sqrt(np.pi)
    return uPDF

#normalised PDF(nu(sigma(M(R))))
def PDF(nu):#(r_min, r_max, k_min, k_max, res): 
    #nu = (1.68647/sig(r_min, r_max, k_min, k_max, res))**2 
    #CDF = integrate.cumtrapz(unnormPDF(r_min, r_max, k_min, k_max, res), nu, initial=0)
    nu=np.linspace(0.1,200,100)
    CDF = integrate.cumtrapz(unnormPDF(nu), nu, initial=0)
    norm = CDF[-1]
    return unnormPDF(nu)/norm

#sampler as a function of nu
def sample_HMF(nu):#(r_min, r_max, k_min, k_max, res, n=None): 
    #nu = (1.68647/sig(r_min, r_max, k_min, k_max, res))**2 
    #CDF = integrate.cumtrapz(unnormPDF(r_min, r_max, k_min, k_max, res), nu, initial=0)
    nu=np.linspace(0.1,200,100)
    CDF = integrate.cumtrapz(unnormPDF(nu), nu, initial=0)
    CDF = CDF/CDF[-1]
    nurand = np.random.uniform(0,1,n)
    sample = np.interp(nurand, CDF, nu)
    return sample
    
    
    
 #-----------------------------------------------------------   
    
    
    
#TEST:
#generate nu using the corresponding function in terms of P(k) and tophat
#vu_values = nu(0.2, 10., 0.01, 10., 100)


nu_values=np.linspace(0.1,200.,100)
plt.plot(nu_values,PDF(nu_values))
plt.xlim([0.01,200])
plt.yscale('log')
plt.xscale('log')
plt.show()


    