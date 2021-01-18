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
#    log (\bar{R}/kpc) = -0.4aM + b,
#
# with :math:`a` and :math:`b` fitting constants. Likewise, for late-type galaxies
# the formula (15) reads
#
# .. math::
#
#    log (\bar{R}/kpc) = -0.4\alpha M +(\beta -\alpha) log \left[1 + 10^{-0.4(M-M_0)}\right] +\gamma
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
#
# .. math::
#
#    p(R) = \frac{1}{\sigma R \sqrt{2 \pi}}e^{- \frac{(ln R - \mu)^2}{2 \sigma^2}}
#
# with mean :math:`\mu = 0` and standard deviation :math:`\sigma`, and an amplitude
# :math:`\bar{R}` given by the Shen et al. 2003 model [1]_ (c.f. equations above).
#
# In this example, the values of the fitting parameters
# are taken from the model in Shen et al. 2003 [1]_.

import numpy as np
import matplotlib.pyplot as plt
from skypy.galaxy import size

# Parameters for the late-type and early-type galaxies
alpha, beta, gamma = 0.21, 0.53, -1.31
a, b = 0.6, -4.63
M0 = -20.52
sigma1, sigma2 = 0., 0.

# Size
m = np.linspace(-16, -24, 100)
slate = size.late_type_lognormal(m, alpha, beta, gamma, M0, sigma1, sigma2)
searly = size.early_type_lognormal(m, a, b, M0, sigma1, sigma2)

# %%
# Validation with the SDSS sample
# -------------------------------
# In this example, we reproduce figure 4 in reference [1]_ and use SDSS data [2]_.
# We obtain the SDSS DR7 sample via `astroquery.sdss` and select the Petrosian
# :math:`R_{50}` and :math:`R_{90}` radii for the r-band magnitudes and redshifts
# of 50000 objects. We then convert radii into physical size for a Planck15
# cosmology. Finally we split the SDSS sample
# using the concentration index, :math:`c`:
#
# .. math::
#
#    c \equiv R_{90} / R_{50}
#
# and the conditions :math:`c < 2.86` for late-type and :math:`c > 2.86` for
# early-type galaxies.


# Load SDSS data release 7
R50_r_phys, c, M_r = np.genfromtxt('SDSS_DR7.csv', delimiter=',')

# Split the sample
M_r_late, R50_r_phys_late = M_r[c < 2.86], np.log10(R50_r_phys[c < 2.86])
M_r_early, R50_r_phys_early = M_r[c > 2.86], np.log10(R50_r_phys[c > 2.86])

# Plot
plt.scatter(M_r_late, R50_r_phys_late, color='lightskyblue', marker='+', alpha=0.01)
plt.scatter(M_r_early, R50_r_phys_early, color='coral',  marker='+', alpha=0.01)
plt.plot(m, np.log10(slate.value), 'b', label='SkyPy Late')
plt.plot(m, np.log10(searly.value), 'r', label='SkyPy Early')

plt.ylim(-1.5, 1.5)
plt.xlabel('$M$')
plt.ylabel('$R_{50,r}$ (kpc)')
plt.title("SDSS data release 7")
plt.legend(frameon=False)

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
# .. [2] https://www.sdss.org
