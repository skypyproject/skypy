"""
Galaxy Ellipticity Distributions
================================

This example demonstrate how to sample ellipticity distributions
in SkyPy.

"""


# %%
# 3D Ellipticity Distribution
# ---------------------------
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
# The func:`skypy.galaxies.morphology.ryden04_ellipticity()` model samples
# projected 2D axis ratios. Specifically, it samples the axis ratios of the
# 3D ellipsoid according to the description above [1] and
# then randomly projects them using triaxial_axis_ratio().
#
#
# Validation plot with SDSS Data
# ------------------------------
#
# Here we validate our function by comparing our ``SkyPy`` simulated galaxy
# ellipticities against observational data from SDSS DR1 in Figure 1 [1].
# You can download the data file
# :download:`SDSS_ellipticity <../../../examples/galaxies/SDSS_ellipticity.txt>`,
# stored as a 2D array: :math:`e_1`, :math:`e_2`.
#

import numpy as np

# Load SDSS data from Fig. 1 in Ryden 2004
data = np.load('SDSS_ellipticity.npz')
e1, e2 = data['sdss']['e1'], data['sdss']['e2']

Ngal = len(e1)
e = np.hypot(e1, e2)
q_amSDSS = np.sqrt((1 - e)/(1 + e))


# %%
#
# In this example, we generate 100 galaxy ellipticity samples using the
# ``SkyPy`` function
# :func:`skypy.galaxies.morphology.ryden04_ellipticity()` and the
# best fit parameters
# given in Ryden 2004 [1]:
# :math:`\mu_\gamma =0.222`, :math:`\sigma_\gamma=0.057`, :math:`\mu =-1.85`
# and :math:`\sigma=0.89`.
#

from skypy.galaxies.morphology import ryden04_ellipticity

# Best fit parameters from Fig. 1 in Ryden 2004
mu_gamma, sigma_gamma, mu, sigma = 0.222, 0.057, -1.85, 0.89

# Binning scheme of Fig. 1
bins = np.linspace(0, 1, 41)
mid = 0.5 * (bins[:-1] + bins[1:])

# Mean and variance of sampling
mean = np.zeros(len(bins)-1)
var = np.zeros(len(bins)-1)

# %%
#

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
#
# We now plot the distribution of axis ratio :math:`𝑞_{am}`
# using adaptive moments in the i band, for exponential galaxies in the SDSS DR1
# (solid line). The data points with error bars represent
# the `SkyPy` simulation:
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
