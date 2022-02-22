"""
Galaxy Demographics
===================
This example demonstrates how to sample any type of galaxy population
from a general Schechter mass function as implemented in SkyPy.
"""

# %%
# Schechter Parameters
# --------------------
# 
# We use the blue parameters in Wiegel et al. 2016 [1]_.
# Also the fraction of satellite galaxies from [1].
# We use a fixed value for the fraction of satellite-quenched galaxies
# :math:`f_{\rho} = 0.5`.

import numpy as np
import matplotlib.pyplot as plt
from skypy.galaxies.stellar_mass import schechter_smf_parameters
from astropy.table import Table

# %%
# Weigel et al 2016 Model
# -----------------------
#
# Here we compare our sampled galaxies.

# %%
# Sonification
# ------------
# STRAUSS clip!



# %%
# References
# ----------
#
#
# .. [1] de la Bella et al. 2021, Quenching and Galaxy Demographics, arXiv 2112.11110.
# 
# .. [2] Weigel 2016
# 
#  .. [3] Trayford J., 2021, james-trayford/strauss: v0.1.0 Pre-release, doi:10.5281/zenodo.5776280, https://doi.org/10.5281/ zenodo.5776280