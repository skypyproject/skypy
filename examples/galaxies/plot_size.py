"""
Galaxy size
===========

This example demonstrate how to obtain sizes for early and late type galaxies
in SkyPy.

"""


# %%
# Size-magnitude relation
# -------------------------
#
# According to Shen et al. 2003 [1]_, the observed size of galaxies
# and the absolute magnitudes follow simple analytic formulae. For early-type
# galaxies, the relation (14) reads
#
# .. math::
#
#    \log_{10} (\bar{R}/{\rm kpc}) = -0.4aM + b,
#
# with :math:`a` and :math:`b` fitting constants. Likewise, for late-type galaxies
# the formula (15) reads
#
# .. math::
#
#    \log_{10}(\bar{R}/{\rm kpc})=-0.4\alpha M+
#                                  (\beta -\alpha)\log \left[1+10^{-0.4(M-M_0)}\right]+\gamma
#
# with a dispersion given by equation 16
#
# .. math::
#
#    \sigma_{ln R} = \sigma_2 + \frac{\sigma_1 - \sigma_2}{1 + 10^{-0.8(M - M_0)}}
#
# where :math:`\alpha`, :math:`\beta`, :math:`\gamma`, :math:`\sigma_1`, :math:`\sigma_2` and
# :math:`M_0` are fitting parameters.
#
# In SkyPy, we draw physical sizes of both galaxy types from a lognormal distribution
# with standard deviation, :math:`\sigma`, and amplitude,
# :math:`\bar{R}`, given by the Shen et al. 2003 model:
# :func:`skypy.galaxy.size.early_type_lognormal()` and
# :func:`skypy.galaxy.size.late_type_lognormal()`
# (c.f. equations above).
#
# In this example, the values of the fitting parameters
# are taken from the model in Shen et al. 2003 [1]_.

import numpy as np
import matplotlib.pyplot as plt
from skypy.galaxy.size import early_type_lognormal, late_type_lognormal

# Parameters for the late-type and early-type galaxies
alpha, beta, gamma = 0.21, 0.53, -1.31
a, b = 0.6, -4.63
M0 = -20.52
sigma1, sigma2 = 0.48, 0.25

# SkyPy late sample
M_late = np.random.uniform(-16, -24, size=10000)
R_late = late_type_lognormal(M_late, alpha, beta, gamma, M0, sigma1, sigma2).value

# SkyPy early sample
M_early = np.random.uniform(-18, -24, size=10000)
R_early = early_type_lognormal(M_early, a, b, M0, sigma1, sigma2).value

# %%
# Validation with the SDSS sample
# -------------------------------
# In this example, we reproduce figure 4 in reference [1]_ .
# The data files can be downloaded here: `early-type
# <https://github.com/skypyproject/skypy/raw/master/examples/galaxies/Shen+03_early.txt>`_
# and `late-type
# <https://github.com/skypyproject/skypy/raw/master/examples/galaxies/Shen+03_late.txt>`_.
# The columns of these files represent magnitudes, radii, lower error and
# upper error.

# Load data from figure 4 in Shen et al 2003
sdss_early = np.loadtxt('Shen+03_early.txt')
sdss_late = np.loadtxt('Shen+03_late.txt')
error_late = sdss_late[:, 3] - sdss_late[:, 2]
error_early = sdss_early[:, 3] - sdss_early[:, 2]

# Bins for median radii
M_bins_late = np.arange(-16, -24.1, -0.5)
M_bins_early = np.arange(-18, -24.1, -0.5)

# Median sizes for SkyPy late- and early-type galaxies
R_bar_early = [np.median(R_early[(M_early <= Ma) & (M_early > Mb)])
               for Ma, Mb in zip(M_bins_early, M_bins_early[1:])]
R_bar_late = [np.median(R_late[(M_late <= Ma) & (M_late > Mb)])
              for Ma, Mb in zip(M_bins_late, M_bins_late[1:])]

# Plot
plt.plot((M_bins_early[:-1]+M_bins_early[1:])/2, R_bar_early, 'r', label='SkyPy early')
plt.plot((M_bins_late[:-1]+M_bins_late[1:])/2, R_bar_late, 'b', label='SkyPy late')

plt.errorbar(sdss_early[:, 0], sdss_early[:, 1], yerr=error_early, color='coral',
             marker='s', label='Shen+03 early', lw=0.5)
plt.errorbar(sdss_late[:, 0], sdss_late[:, 1], yerr=error_late, color='deepskyblue',
             marker='^', label='Shen+03 late', lw=0.5)

plt.ylim(5e-1, 2e1)
plt.xlim(-24, -15.5)
plt.xlabel('Magnitude $M$')
plt.ylabel('$R_{50,r} (kpc)$')
plt.title("SDSS data release 7")
plt.legend(frameon=False)

plt.yscale('log')
plt.show()

# %%
# References
# ----------
#
#
# .. [1] S. Shen, H.J. Mo, S.D.M. White, M.R. Blanton, G. Kauffmann, W. Voges,
#   Brinkmann, I. Csabai, `Mon. Not. Roy. Astron. Soc. 343, 978 (2003)`_
#
# .. _Mon. Not. Roy. Astron. Soc. 343, 978 (2003): https://arxiv.org/pdf/astro-ph/0301527.pdf
