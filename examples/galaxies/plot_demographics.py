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

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.units import Quantity
from skypy.galaxies import schechter_smf
# from skypy.galaxies.stellar_mass import (schechter_smf_amplitude_centrals,
                                  #    schechter_smf_amplitude_satellites,
                                #    schechter_smf_amplitude_mass_quenched,
                             #    schechter_smf_amplitude_satellite_quenched
#                                          )

# %%


# Weigel et al. 2016 parameters for the active population
phiblue = 10**-2.423
mstarb = 10**10.60
alphab = -1.21
blue = (phiblue, alphab, mstarb)

# Fraction of satellite-quenched galaxies and satellites
frho = 0.5
fsat = 0.4

# Weigel+16 redshift and cosmology
z_min, z_max = 0.02, 0.06
z_range = np.linspace(z_min, z_max, 100)
cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

# Sky area (SDSS DR7 8423 deg2)
sky_area = Quantity(2000, "deg2")

# %%
# First we use the model implemented in `skypy.galaxies.stellar_mass`.


# To be replaced by the SkyPy function once it's merged
def schechter_smf_phi_centrals(phi_blue_total, fsatellite):
    if np.ndim(phi_blue_total) == 1 and np.ndim(fsatellite) == 1:
        phi_blue_total = phi_blue_total[:, np.newaxis]

    sum_phics = (1 - fsatellite) * (1 - np.log(1 - fsatellite))
    return (1 - fsatellite) * phi_blue_total / sum_phics


def schechter_smf_phi_satellites(phi_centrals, fsatellite):
    return phi_centrals * np.log(1 / (1 - fsatellite))


def schechter_smf_phi_mass_quenched(phi_centrals, phi_satellites):
    return phi_centrals + phi_satellites


def schechter_smf_phi_satellite_quenched(phi_satellites, fenvironment):
    return - np.log(1 - fenvironment) * phi_satellites


# SkyPy amplitudes
phic = schechter_smf_phi_centrals(phiblue, fsat)
phis = schechter_smf_phi_satellites(phic, fsat)
phimq = schechter_smf_phi_mass_quenched(phic, phis)
phisq = schechter_smf_phi_satellite_quenched(phis, frho)


# %%
# Finally we simulate the galaxy populations.


# Schechter mass functions
z_centrals, m_centrals = schechter_smf(z_range, mstarb, phic, alphab, 1e9, 1e12, sky_area, cosmology)
z_satellites, m_satellites = schechter_smf(z_range, mstarb, phis, alphab, 1e9, 1e12, sky_area, cosmology)
z_massq, m_mass_quenched = schechter_smf(z_range, mstarb, phimq, alphab + 1, 1e9, 1e12, sky_area, cosmology)
z_satq, m_satellite_quenched = schechter_smf(z_range, mstarb, phisq, alphab, 1e9, 1e12, sky_area, cosmology)


logm_centrals = np.log10(m_centrals)
logm_satellites = np.log10(m_satellites)
logm_massq = np.log10(m_mass_quenched)
logm_satq = np.log10(m_satellite_quenched)

# log Mass bins
bins = np.linspace(9, 12, 35)

# Sky volume
dV_dz = (cosmology.differential_comoving_volume(z_range) * sky_area).to_value('Mpc3')
dV = np.trapz(dV_dz, z_range)
dlm = (np.max(bins)-np.min(bins)) / (np.size(bins)-1)

# log distribution of masses
logphi_centrals = np.histogram(logm_centrals, bins=bins)[0] / dV / dlm
logphi_satellites = np.histogram(logm_satellites, bins=bins)[0] / dV / dlm
logphi_massq = np.histogram(logm_massq, bins=bins)[0] / dV / dlm
logphi_satq = np.histogram(logm_satq, bins=bins)[0] / dV / dlm

logphi_active = logphi_centrals + logphi_satellites
logphi_passive = logphi_massq + logphi_satq
logphi_total = logphi_active + logphi_passive


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


# Load the data
wtotal = Table.read('weigel16_total.csv', format='csv')
wred = Table.read('weigel16_quiescent.csv', format='csv')
wblue = Table.read('weigel16_active.csv', format='csv')
wcentral = Table.read('weigel16_central.csv', format='csv')
wsatellite = Table.read('weigel16_satellite.csv', format='csv')

# %%


# Plot
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), sharex=True, sharey=True)
fig.suptitle('Galaxy Demographics Simulation', fontsize=26)

ax1.plot(wblue['logm'], 10**wblue['logphi'], color='k', label='Weigel+16 active', lw=1)
ax1.step(bins[:-1], logphi_active, where='post', label='SkyPy active', color='blue', zorder=3, lw=1)
ax1.step(bins[:-1], logphi_centrals, where='post', label='SkyPy centrals', color='royalblue', zorder=3, lw=1)
ax1.step(bins[:-1], logphi_satellites, where='post', label='SkyPy satellites', color='cyan', zorder=3, lw=1)


ax2.plot(wred['logm'], 10**wred['logphi'], color='k', label='Weigel+16 passive', lw=1)
ax2.fill_between(wred['logm'], 10**wred['upper_error'], 10**wred['lower_error'], color='salmon', alpha=0.1)
ax2.step(bins[:-1], logphi_passive, where='post', label='SkyPy passive', color='red', zorder=3, lw=1)
ax2.step(bins[:-1], logphi_massq, where='post', label='SkyPy mass-quenched', color='coral', zorder=3, lw=1)
ax2.step(bins[:-1], logphi_satq, where='post', label='SkyPy sat-quenched', color='maroon', zorder=3, lw=1)

ax3.plot(wtotal['logm'], 10**wtotal['logphi'], color='k', label='Weigel+16 total', lw=1)
ax3.plot(wcentral['logm'], 10**wcentral['logphi'], '--', color='grey', label='Weigel+16 centrals', lw=1)
ax3.plot(wsatellite['logm'], 10**wsatellite['logphi'], '--', color='grey', label='Weigel+16 satellites', lw=1)
ax3.fill_between(wtotal['logm'], 10**wtotal['upper_error'], 10**wtotal['lower_error'], color='plum', alpha=0.1)
ax3.step(bins[:-1], logphi_total, where='post', label='SkyPy total', color='purple', zorder=3, lw=1)

for ax in [ax1, ax2, ax3]:
    ax.legend(loc='lower left', fontsize='small', frameon=False)
    ax.set_xlabel(r'Stellar mass, $log (M/M_{\odot})$', fontsize=18)
    ax.set_xlim((9, 11.9))
    ax.set_ylim((2e-6,5e-2))
    ax.set_yscale('log')


ax1.set_ylabel(r'$\phi /h^3 Mpc^{-3}$', fontsize=18)
# plt.savefig('galaxy_simulation.pdf')
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
IPython.display.Audio("skypy_sim_active.wav")

# %%

# Mass-quenched galaxies
IPython.display.Audio("skypy_sim_massq.wav")

# %%

# Satellite-quenched galaxies
IPython.display.Audio("skypy_sim_satq.wav")


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
