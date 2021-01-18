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


# Load SDSS data release 7
R50_r_phys, c, M_r = np.genfromtxt('SDSS_DR7.csv', delimiter=',')

# Parameters for the late-type and early-type galaxies
alpha, beta, gamma = 0.21, 0.53, -1.31
a, b = 0.6, -4.63
M0 = -20.52
sigma1, sigma2 = 0., 0.

# Size
m = np.linspace(-16, -24, 100)
slate = size.late_type_lognormal(m, alpha, beta, gamma, M0, sigma1, sigma2)
searly = size.early_type_lognormal(m, a, b, M0, sigma1, sigma2)


# Split data: c < 2.86 late type, c > 2.86 early type
M_r_late, R50_r_phys_late = M_r[c < 2.86], np.log10(R50_r_phys[c < 2.86])
M_r_early, R50_r_phys_early = M_r[c > 2.86], np.log10(R50_r_phys[c > 2.86])

plt.scatter(M_r_late, R50_r_phys_late, color='lightskyblue', marker='+', alpha=0.01)
plt.scatter(M_r_early, R50_r_phys_early, color='coral',  marker='+', alpha=0.01)

plt.plot(m, np.log10(slate.value), 'b', label='SkyPy Late')
plt.plot(m, np.log10(searly.value), 'r', label='SkyPy Early')

# plt.yscale('log')
plt.xlabel('$M$')
plt.ylabel('$R_{50,r}$ (kpc)')
plt.legend(frameon=False)
plt.title("SDSS data release 7")

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
