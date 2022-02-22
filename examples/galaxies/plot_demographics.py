"""
Galaxy Demographics
===================
This example demonstrates how to sample any type of galaxy population
from a general Schechter mass function as implemented in SkyPy.
"""

# %%
# Schechter Parameters
# --------------------
# 
# We use the blue parameters in Wiegel et al. 2016 [1]_.
# Also the fraction of satellite galaxies from [1].
# We use a fixed value for the fraction of satellite-quenched galaxies
# :math:`f_{\rho} = 0.5`.

import numpy as np
import matplotlib.pyplot as plt
# from skypy.galaxies.stellar_mass import schechter_smf_parameters
from astropy.table import Table

# Replace by the SkyPy function once it's merged

def schechter_smf_parameters(active_parameters, fsatellite, fenvironment):
    phi, alpha, mstar = active_parameters
    
    sum_phics = (1 - fsatellite) * (1 -  np.log(1 - fsatellite))
    
    phic = (1 - fsatellite) * phi / sum_phics
    phis = phic * np.log(1 / (1 - fsatellite))
    
    centrals = (phic, alpha, mstar)
    satellites = (phis, alpha, mstar)
    mass_quenched = (phic + phis, alpha + 1, mstar)
    satellite_quenched = (- np.log(1 - fenvironment) * phis, alpha, mstar)
    
    return {'centrals': centrals, 'satellites': satellites,
            'mass_quenched': mass_quenched, 'satellite_quenched': satellite_quenched}


# Weigel et al. 2016 parameters for the active population
phiblue = 10**-2.423
mstarb = 10**10.60
alphab = -1.21
blue_params = (phiblue, alphab, mstarb)

# Choose a fraction of satellite-quenched galaxies
frho = 0.5

# Compute the fraction of satellite galaxies
wtotal = Table.read('weigel16_total.csv', format='csv')
wsatellite = Table.read('weigel16_satellite.csv', format='csv')
logm = wtotal['logm']
fsat = 10**wsatellite['logphi']/10**wtotal['logphi']

# Generate the Schechter parameters for all populations
sp = schechter_smf_parameters(blue_params, fsat, frho)

# Compute the Schechter mass functions for all populations
# SMF ideally from SkyPy
def schechter_dndm(mass, params):
    phi, alpha, mstar = params
    x = mass / mstar
    return phi * x**alpha * np.exp(-x)

m = 10**logm
gb = schechter_dndm(m, blue_params)
gc = schechter_dndm(m, sp['centrals'])
gs = schechter_dndm(m, sp['satellites'])
gmq = schechter_dndm(m, sp['mass_quenched'])
gsq = schechter_dndm(m, sp['satellite_quenched'])

active = gc + gs
passive = gmq + gsq
total = active + passive 

# %%
# Weigel et al 2016 Model
# -----------------------
#
# Here we compare our sampled galaxies.

# %%
# Sonification
# ------------
# STRAUSS clip!



# %%
# References
# ----------
#
#
# .. [1] de la Bella et al. 2021, Quenching and Galaxy Demographics, arXiv 2112.11110.
# 
# .. [2] Weigel 2016
# 
#  .. [3] Trayford J., 2021, james-trayford/strauss: v0.1.0 Pre-release, doi:10.5281/zenodo.5776280, https://doi.org/10.5281/ zenodo.5776280