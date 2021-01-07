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

# Variance
sigma_lnR = sigma2 + (sigma1 - sigma2) / (1.0 + np.power(10, -0.8 * (mag - M0)))

fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True, figsize=(10,4))
ax0.plot(np.flip(mag), slate, 'r.', label='Late')
ax0.plot(np.flip(mag), searly, 'b.', label='Early')

ax0.plot(np.flip(mag), rlate, 'r--')
ax0.plot(np.flip(mag), rearly, 'b--')

ax0.set_yscale('log')
ax0.set_xlabel('M')
ax0.set_ylabel('R (kpc)')
ax0.legend(frameon=False)

ax1.plot(np.flip(mag), sigma_lnR)

ax1.set_xlabel('M')
ax1.set_ylabel(r'$\sigma_{lnR}$ (kpc)')

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
