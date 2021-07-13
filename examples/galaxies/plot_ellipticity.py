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
# In Ryden 2004 [1]_, ...
#
# .. math::
#
#    \log_{10} (\bar{R}/{\rm kpc}) = -0.4aM + b,
#
# with :math:`a` and :math:`b` fitting constants. Likewise, late-type galaxies
# follow Equation 15:
#

import numpy as np
import matplotlib.pyplot as plt
from skypy.galaxies.morphology import ryden04_ellipticity

# %%
# Validation against SDSS Data
# ----------------------------

# Plot here

# %%
# References
# ----------
#
#
# .. [1] Ryden, Barbara S., 2004, `The Astrophysical Journal, Volume 601, Issue 1, pp. 214-220`_
#
# .. _The Astrophysical Journal, Volume 601, Issue 1, pp. 214-220: https://arxiv.org/abs/astro-ph/0310097
