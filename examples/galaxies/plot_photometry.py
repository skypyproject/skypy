"""
Optical Photometry
==================

This example demonstrates...

"""

# %%
# kcorrect Spectral Templates
# ---------------------------
#
# Describe kcorrect templates [1]_
#
# .. literalinclude:: ../../../examples/galaxies/mccl_galaxies.yml
#   :language: YAML

# %%
# SDSS Photometry
# ---------------
#
# Here we compare with SDSS data [2]_

from astropy.table import Table, vstack
from matplotlib import pyplot as plt
import numpy as np
from skypy.pipeline import Pipeline

# Execute SkyPy galaxy photometry simulation pipeline
pipeline = Pipeline.read("mccl_galaxies.yml")
pipeline.execute()
skypy_galaxies = vstack([pipeline['blue_galaxies'], pipeline['red_galaxies']])

# Read SDSS magnitude distributions for a 1 degree^2
sdss_data = Table.read("sdss_dered_10deg2.ecsv", format='ascii.ecsv')

# Plot magnitude distributions for SkyPy simulation and SDSS data
bins = np.linspace(14.95, 25.05, 102)
plt.hist(skypy_galaxies['mag_r'], bins=bins, alpha=0.5, color='r', label='SkyPy-r')
plt.hist(skypy_galaxies['mag_u'], bins=bins, alpha=0.5, color='b', label='SkyPy-u')
plt.plot(sdss_data['magnitude'], sdss_data['dered_r'], color='r', label='SDSS-r')
plt.plot(sdss_data['magnitude'], sdss_data['dered_u'], color='b', label='SDSS-u')
plt.xlim(16, 24)
plt.yscale('log')
plt.xlabel(r'$\mathrm{Apparent\,Magnitude}$')
plt.ylabel(r'$\mathrm{N} \, [\mathrm{deg}^{-2} \, \mathrm{mag}^{-1}]$')
plt.legend()
plt.show()

# %%
# References
# ----------
#
# .. [1] M. R. Blanton and S. Roweis, 2007, AJ, 125, 2348
# .. [2] K. N. Abazajian et al. 2009, ApJS, 182, 543
#
