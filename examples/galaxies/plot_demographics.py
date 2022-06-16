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
# We use the blue parameters in Wiegel et al. 2016 [2]_.
# Also the fraction of satellite galaxies from [2].
# We use a fixed value for the fraction of satellite-quenched galaxies
# :math:`f_{\rho} = 0.5`.

import numpy as np
import matplotlib.pyplot as plt
# from skypy.galaxies.stellar_mass import (schechter_smf_amplitude_centrals,
                                  #    schechter_smf_amplitude_satellites,
                                #    schechter_smf_amplitude_mass_quenched,
                             #    schechter_smf_amplitude_satellite_quenched
#                                          )
from astropy.table import Table


# Replace by the SkyPy function once it's merged
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

# Choose a fraction of satellite-quenched galaxies
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


# Compute the Schechter mass functions for all populations
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
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True)
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
# STRAUSS clip! [3]_.


# %%
# References
# ----------
#
#
# .. [1] de la Bella et al. 2021, Quenching and Galaxy Demographics, arXiv 2112.11110.
#
# .. [2] Weigel A. K., Schawinski K., Bruderer C., 2016, Monthly Notices of the
# Royal Astronomical Society, 459, 2150
#
# .. [3] Trayford J., 2021, james-trayford/strauss: v0.1.0 Pre-release, doi:10.5281/zenodo.5776280,
# https://doi.org/10.5281/ zenodo.5776280
