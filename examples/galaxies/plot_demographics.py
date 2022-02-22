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
# Validation against SSD DR7 data
# -------------------------------
# 
# Weigel et al 2016 Model.
# Here we compare our sampled galaxies.

# Load the rest of data
wred = Table.read('weigel16_quiescent.csv', format='csv')
wblue = Table.read('weigel16_active.csv', format='csv')
wcentral = Table.read('weigel16_central.csv', format='csv')

# This factor allows us to go from :math:`\phi` to a :math: `\log \phi` plot
# and compare with Weigel et al 2016 best-fit model
factor = np.log(10) * 10**logm / mstarb
lblue, lcentrals, lsats = np.log10(gb * factor), np.log10(gc * factor), np.log10(gs * factor)
lred, lmq, lsq = np.log10(passive * factor), np.log10(gmq * factor), np.log10(gsq * factor)
ltotal = np.log10(total * factor)

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,6), sharex=True, sharey=True)
fig.suptitle('Galaxy Demographics', fontsize=26)

ax1.plot(wblue['logm'], wblue['logphi'], color='k', label='Weigel+16', lw=1)
ax1.plot(logm, lblue, color='blue', label='SkyPy Active', lw=1)
ax1.plot(logm, lcentrals, '--', color='royalblue', label='SkyPy Centrals', lw=1)
ax1.plot(logm, lsats, '--', color='cyan', label='SkyPy Satellites', lw=1)

ax2.plot(wred['logm'], wred['logphi'], color='k', label='Weigel+16', lw=1)
ax2.fill_between(wred['logm'], wred['upper_error'], wred['lower_error'], color='salmon', alpha=0.1)
ax2.plot(logm, lred, color='red', label='SkyPy Passive', lw=1)
ax2.plot(logm, lmq, '--', color='coral', label='SkyPy MassQ', lw=1)
ax2.plot(logm, lsq, '--', color='maroon', label='SkyPy SatQ', lw=1)

ax3.plot(wtotal['logm'], wtotal['logphi'], color='k', label='Weigel+16', lw=1)
ax3.plot(wcentral['logm'], wcentral['logphi'], '--', color='grey', label='Centrals', lw=1)
ax3.plot(wsatellite['logm'], wsatellite['logphi'], '--', color='grey', label='Satellites', lw=1)
ax3.fill_between(wtotal['logm'], wtotal['upper_error'], wtotal['lower_error'], color='plum', alpha=0.1)
ax3.plot(logm, ltotal, color='purple', label='SkyPy Total', lw=1)


for ax in [ax1, ax2,ax3]:
    ax.legend(loc='lower left', frameon=False, fontsize=14)
    ax.set_xlabel(r'Stellar mass, $log (M/M_{\odot})$', fontsize=18)
    ax.set_ylim(-5.5)


ax1.set_ylabel(r'$log(\phi /h^3 Mpc^{-3}dex^{-1} )$', fontsize=18)
plt.tight_layout()
plt.show()

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