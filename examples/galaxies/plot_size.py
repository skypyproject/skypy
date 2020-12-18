"""
Galaxy size
===========

This example demonstrate how to obtain sizes for early and late type galaxies
in SkyPy.

"""


# %%
# Galaxy sizes
# ------------
#
# Add some notes here. We are trying to reporduce figure 6 in
# reference [1]_ and use data from SDSS.

import numpy as np
import matplotlib.pyplot as plt
from skypy.galaxy import size

mag = np.linspace(-16, -24)

# Parameters for the late-type galaxies
alpha, beta, gamma, M0 = 0.21, 0.53, -1.31, -20.52
sigma1, sigma2 = 0.48, 0.25

# Parameters for the early-tyoe galaxies
a, b, M0 = 0.6, -4.63, -20.52
sigma1, sigma2 = 0.48, 0.25

# Size
slate = size.late_type_lognormal(mag, alpha, beta, gamma, M0, sigma1, sigma2)
searly = size.early_type_lognormal(mag, a, b, M0, sigma1, sigma2)

# Mean radius
rlate = np.power(10, -0.4 * alpha * mag + (beta - alpha) *
                 np.log10(1 + np.power(10, -0.4 * (mag - M0))) + gamma)
rearly = np.power(10, -0.4 * a * mag + (a - a) *
                  np.log10(1 + np.power(10, -0.4 * (mag - M0))) + b)

plt.plot(np.flip(mag), slate, 'r.', label='Late')
plt.plot(np.flip(mag), searly, 'b.', label='Early')

plt.plot(np.flip(mag), rlate, 'r--')
plt.plot(np.flip(mag), rearly, 'b--')

plt.yscale('log')
plt.xlabel('M')
plt.ylabel('R (kpc)')

plt.legend(frameon=False)
plt.show()

# Variance
sigma_lnR = sigma2 + (sigma1 - sigma2) / (1.0 + np.power(10, -0.8 * (mag - M0)))

plt.plot(np.flip(mag), sigma_lnR)

plt.xlabel('M')
plt.ylabel(r'$\sigma_{lnR}$ (kpc)')

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
