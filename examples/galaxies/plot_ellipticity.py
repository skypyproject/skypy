"""
Galaxy Ellipticity Distributions
================================

This example demonstrate how to sample ellipticity distributions
in SkyPy.

"""


# %%
# 3D ellipticity istribution
# --------------------------
#
# In Ryden 2004 [1]_, the ellipticity is sampled by randomly projecting
# a 3D ellipsoid with principal axes :math:`A > B > C`.
#
# The distribution of the axis ratio :math:`\gamma = C/A` is a truncated
# normal with mean :math:`\mu_\gamma` and standard deviation
# :math:`\sigma_\gamma`.
#
# The distribution of :math:`\epsilon = \log(1 - B/A)` is truncated normal
# with mean :math:`\mu` and standard deviation :math:`\sigma`.
#
#
# Ellipticity SDSS Data
# ---------------------
#
# Here we reproduce Figure 1 from [1]_, comparing our simulated galaxy
# ellipticities against observational data from SDSS DR1.
# You can download the data file
# :download:`SDSS_ellipticity <../../../examples/galaxies/SDSS_ellipticity.txt>`.
# stored as a 2D array: ellipticity_X, ellipticity_T.
#
# The best fit parameters [1]_:
# :math:`\mu_\gamma, \sigma_\gamma, \mu, \sigma = 0.222, 0.057, -1.85, 0.89`.

import numpy as np

# Load SDSS data from Fig. 1 in Ryden 2004
eX, eT = np.loadtxt('SDSS_ellipticity.txt', unpack=True)
Ngal = len(eX)
e = np.hypot(eX, eT)
q_amSDSS = np.sqrt((1 - e)/(1 + e))

# Best fit parameters from Fig. 1 in Ryden 2004
mu_gamma, sigma_gamma, mu, sigma = 0.222, 0.057, -1.85, 0.89


# %%
# SkyPy Ellipticity model
# -----------------------
#
# We use the Ryden04 model in SkyPy.

from skypy.galaxies.morphology import ryden04_ellipticity

# Binning scheme of Fig. 1
bins = np.linspace(0, 1, 41)
mid = 0.5 * (bins[:-1] + bins[1:])

# Mean and variance of sampling
mean = np.zeros(len(bins)-1)
var = np.zeros(len(bins)-1)

# Create 100 SkyPy realisations
for i in range(100):
    # sample ellipticity
    e = ryden04_ellipticity(mu_gamma, sigma_gamma, mu, sigma, size=Ngal)
    # recover axis ratio from ellipticity
    q = (1 - e)/(1 + e)
    # bin
    n, _ = np.histogram(q, bins=bins)

    # update mean and variance
    x = n - mean
    mean += x/(i+1)
    y = n - mean
    var += x*y

# finalise variance and standard deviation
var = var/i
std = np.sqrt(var)

# %%
# Plot
# ----
#

import matplotlib.pyplot as plt

plt.hist(q_amSDSS, range=[0, 1], bins=40, histtype='step',
         ec='k', lw=0.5, label='SDSS data')
plt.errorbar(mid, mean, yerr=std, fmt='.r', ms=4, capsize=3,
             lw=0.5, mew=0.5, label='SkyPy model')

plt.xlabel(r'Axis ratio, $q_{am}$')
plt.ylabel(r'N')
plt.legend(frameon=False)
plt.show()

# %%
# References
# ----------
#
#
# .. [1] Ryden, Barbara S., 2004, `The Astrophysical Journal, Volume 601, Issue 1, pp. 214-220`_
#
# .. _The Astrophysical Journal, Volume 601, Issue 1, pp. 214-220: https://arxiv.org/abs/astro-ph/0310097
