"""
Galaxy Demographics
===================

In this example we reproduce the results from de la Bella et al. 2021 [1]_.
In that paper the authors distinguished between active galaxies
(centrals and satellites) and quiescent galaxies (mass-quenched and satellite-quenched),
describing the galaxy demographics with a set of continuity equations that are solved analytically.
Such equations invoke two quenching mechanisms that transform star-forming galaxies into quiescent
objects: mass quenching and satellite quenching.

They showed that the combination of the two quenching mechanisms produces
a double Schechter function for the quiescent population.
They demonstrated that the satellite-quenched galaxies are indeed a subset
of the active galaxies and that the mass-quenched galaxies have a different
faint-end slope parameter by one unit. The connection between quenching and
galaxy populations reduced significantly the parameter space of the simulations.

This example uses de la Bella et al. model implemented in `skypy.galaxies.stellar_mass`
and shows how to sample any type of galaxy population
from a general Schechter mass function.
"""

# %%
# Schechter Parameters
# --------------------
#
# We generate their Figure 4 and use the best-fit values of the blue parameters
# in Wiegel et al. 2016 [2]_. We follow de la Bella et al.'s procedure
# to compute the fraction of satellite galaxies from [2]
# and use their fixed value for the fraction of satellite-quenched galaxies
# :math:`f_{\rho} = 0.5`.
# 
# Finally we generate the different galaxy samples by feeding the Schechter parameters
# as inputs for the Schechter mass function.

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
# from skypy.galaxies.stellar_mass import (schechter_smf_amplitude_centrals,
                                  #    schechter_smf_amplitude_satellites,
                                #    schechter_smf_amplitude_mass_quenched,
                             #    schechter_smf_amplitude_satellite_quenched
#                                          )

# %%
# First we use the model implemented in `skypy.galaxies.stellar_mass`.


# To be replaced by the SkyPy function once it's merged
def schechter_smf_amplitude_centrals(phi_blue_total, fsatellite):
    if np.ndim(phi_blue_total) == 1 and np.ndim(fsatellite) == 1:
        phi_blue_total = phi_blue_total[:, np.newaxis]

    sum_phics = (1 - fsatellite) * (1 - np.log(1 - fsatellite))
    return (1 - fsatellite) * phi_blue_total / sum_phics


def schechter_smf_amplitude_satellites(phi_centrals, fsatellite):
    return phi_centrals * np.log(1 / (1 - fsatellite))


def schechter_smf_amplitude_mass_quenched(phi_centrals, phi_satellites):
    return phi_centrals + phi_satellites


def schechter_smf_amplitude_satellite_quenched(phi_satellites, fenvironment):
    return - np.log(1 - fenvironment) * phi_satellites


# Weigel et al. 2016 parameters for the active population
phiblue = 10**-2.423
mstarb = 10**10.60
alphab = -1.21
blue = (phiblue, alphab, mstarb)

# Fraction of satellite-quenched galaxies
frho = 0.5

# Compute the fraction of satellite galaxies
wtotal = Table.read('weigel16_total.csv', format='csv')
wsatellite = Table.read('weigel16_satellite.csv', format='csv')
logm = wtotal['logm']
fsat = 10**wsatellite['logphi']/10**wtotal['logphi']

# Generate the Schechter parameters for all populations
phic = schechter_smf_amplitude_centrals(phiblue, fsat)
phis = schechter_smf_amplitude_satellites(phic, fsat)
phimq = schechter_smf_amplitude_mass_quenched(phic, phis)
phisq = schechter_smf_amplitude_satellite_quenched(phis, frho)

central = (phic, alphab, mstarb)
satellite = (phis, alphab, mstarb)
mass_quenched = (phimq, alphab + 1, mstarb)
sat_quenched = (phisq, alphab, mstarb)


# %%
# Finally we compute the Schechter mass functions for all populations.


# SMF ideally from SkyPy
def schechter_dndm(mass, params):
    phi, alpha, mstar = params
    x = mass / mstar
    return phi * x**alpha * np.exp(-x)


m = 10**logm
gb = schechter_dndm(m, blue)
gc = schechter_dndm(m, central)
gs = schechter_dndm(m, satellite)
gmq = schechter_dndm(m, mass_quenched)
gsq = schechter_dndm(m, sat_quenched)

active = gc + gs
passive = gmq + gsq
total = active + passive

# %%
# Validation against SSD DR7 data
# -------------------------------
#
# Here we compare our sampled galaxies and
# we validate the model using the results from the best fit to
# SDSS DR7 data in Weigel et al. (2016) [2].
# The authors presented a comprehensive method to determine
# stellar mass functions and apply it to samples in the local universe,
# in particular to SDSS DR7 data in the redshift range from 0.02 to 0.06.
#
# Their data is presented as logarithmic distribution, therefore,
# to perform the comparison we need to apply a conversion factor.
# This factor allows us to go from :math:`\phi` to a :math:`\log \phi` plot
# and compare with Weigel et al 2016 best-fit model. 


# Load the rest of data
wred = Table.read('weigel16_quiescent.csv', format='csv')
wblue = Table.read('weigel16_active.csv', format='csv')
wcentral = Table.read('weigel16_central.csv', format='csv')

# Conversion factor to log distribution
factor = np.log(10) * 10**logm / mstarb
lblue, lcentrals, lsats = np.log10(gb * factor), np.log10(gc * factor), np.log10(gs * factor)
lred, lmq, lsq = np.log10(passive * factor), np.log10(gmq * factor), np.log10(gsq * factor)
ltotal = np.log10(total * factor)

# %%


# Plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True)
fig.suptitle('Galaxy Demographics', fontsize=26)

ax1.plot(wblue['logm'], wblue['logphi'], color='k', label='Weigel+16', lw=1)
ax1.plot(logm, lblue, color='blue', label='SkyPy Active', lw=1)
ax1.plot(logm, lcentrals, '--', color='royalblue', label='SkyPy Centrals', lw=1)
ax1.plot(logm, lsats, '--', color='cyan', label='SkyPy Satellites', lw=1)

ax2.plot(wred['logm'], wred['logphi'], color='k', label='Weigel+16', lw=1)
ax2.fill_between(wred['logm'], wred['upper_error'], wred['lower_error'], color='salmon', alpha=0.1)
ax2.plot(logm, lred, color='red', label='SkyPy Passive', lw=1)
ax2.plot(logm, lmq, '--', color='coral', label='SkyPy Mass Quenched', lw=1)
ax2.plot(logm, lsq, '--', color='maroon', label='SkyPy Sat Quenched', lw=1)

ax3.plot(wtotal['logm'], wtotal['logphi'], color='k', label='Weigel+16', lw=1)
ax3.plot(wcentral['logm'], wcentral['logphi'], '--', color='grey', label='Centrals', lw=1)
ax3.plot(wsatellite['logm'], wsatellite['logphi'], '--', color='grey', label='Satellites', lw=1)
ax3.fill_between(wtotal['logm'], wtotal['upper_error'], wtotal['lower_error'], color='plum', alpha=0.1)
ax3.plot(logm, ltotal, color='purple', label='SkyPy Total', lw=1)


for ax in [ax1, ax2, ax3]:
    ax.legend(loc='lower left', frameon=False, fontsize=14)
    ax.set_xlabel(r'Stellar mass, $log (M/M_{\odot})$', fontsize=18)
    ax.set_ylim(-5.5)


ax1.set_ylabel(r'$log(\phi /h^3 Mpc^{-3}dex^{-1} )$', fontsize=18)
plt.tight_layout()
plt.show()

# %%
# Sonification
# ------------
# The sonification, or transformation of physical data via sound,
# is becoming increasingly important to make astronomy accessible
# for those who are visually impaired, and to enhance visualisations
# and convey information that visualisation alone cannot.
# In this work [1] the authors also made their main plot available
# in sound format using the `STRAUSS`_ software (Trayford 2021) [3]_.

import IPython
# Active population
IPython.display.Audio("skypy_active.wav")

# %%

# Mass-quenched galaxies
IPython.display.Audio("skypy_mass_quenched.wav")

# %%

# Satellite-quenched galaxies
IPython.display.Audio("skypy_satellite_quenched.wav")


# %%
# References
# ----------
#
# .. [1] de la Bella et al. 2021, Quenching and Galaxy Demographics, arXiv 2112.11110.
#
# .. [2] Weigel A. K., Schawinski K., Bruderer C., 2016, Monthly Notices of the
#   Royal Astronomical Society, 459, 2150
#
# .. [3] Trayford J., 2021, james-trayford/strauss: v0.1.0 Pre-release, `doi:10.5281/zenodo.5776280`_.
#   
# .. _doi:10.5281/zenodo.5776280: https://doi.org/10.5281/zenodo.5776280
# 
# .. _STRAUSS: https://strauss.readthedocs.io/en/latest/
