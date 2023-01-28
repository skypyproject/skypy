"""
Optical Photometry
==================

This example demonstrates how to model galaxy photometric magnitudes using the
kcorrect spectral energy distribution templates as implemented in SkyPy.

"""

# %%
# kcorrect Spectral Templates
# ---------------------------
#
# In SkyPy, the rest-frame spectral energy distributions (SEDs) of galaxies can
# be modelled as a linear combination of the five kcorrect basis templates
# [1]_. One possible model for the coefficients is a redshift-dependent
# Dirichlet distribution [2]_ which can be sampled from using the
# :func:`dirichlet_coefficients <skypy.galaxies.spectrum.dirichlet_coefficients>`
# function. The coefficients are then taken by the :meth:`kcorrect.absolute_magnitudes
# <skypy.utils.photometry.SpectrumTemplates.absolute_magnitudes>` and
# :meth:`kcorrect.apparent_magnitudes
# <skypy.utils.photometry.SpectrumTemplates.apparent_magnitudes>`
# methods to calculate the relevant photometric quantities using the
# :doc:`speclite <speclite:overview>` package. Note that since the kcorrect
# templates are defined per unit stellar mass, the total stellar mass of each
# galaxy must either be given or calculated from its absolute magnitude in
# another band using :meth:`kcorrect.stellar_mass
# <skypy.galaxies.spectrum.KCorrectTemplates.stellar_mass>`.
# An example simulation for the SDSS u- and r-band apparent magnitudes of "red"
# and "blue" galaxy populations is given by the following config file:
#
# .. literalinclude:: ../../../examples/galaxies/sdss_photometry.yml
#   :language: YAML
#   :caption: examples/galaxies/sdss_photometry.yml
#
# The config file can be downloaded
# :download:`here <../../../examples/galaxies/sdss_photometry.yml>`
# and the simulation can be run either from the command line and saved to FITS
# files:
#
# .. code-block:: bash
#
#    $ skypy examples/galaxies/sdss_photometry.yml sdss_photometry.fits
#
# or in a python script using the :class:`Pipeline <skypy.pipeline.Pipeline>`
# class as demonstrated in the `SDSS Photometry`_ section below. For more
# details on writing config files see the :doc:`Pipeline Documentation </pipeline/index>`.
#
# SDSS Photometry
# ---------------
#
# Here we compare the apparent magnitude distributions of our simulated
# galaxies with data from a :math:`10 \, \mathrm{deg^2}` region of the Sloan
# Digital Sky Survey [3]_. The binned SDSS magnitude distributions were
# generated from a query of the DR7 data release and can be downloaded
# :download:`here <../../../examples/galaxies/sdss_dered_10deg2.ecsv>`.

from astropy.table import Table, vstack
from matplotlib import pyplot as plt
import numpy as np
from skypy.pipeline import Pipeline

# Execute SkyPy galaxy photometry simulation pipeline
pipeline = Pipeline.read("sdss_photometry.yml")
pipeline.execute()
skypy_galaxies = vstack([pipeline['blue_galaxies'], pipeline['red_galaxies']])

# SDSS magnitude distributions for a 10 degree^2 region
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
# .. [2] J. Herbel, T. Kacprzak, A. Amara, A. Refregier, C.Bruderer and
#    A. Nicola 2017, JCAP, 1708, 035
# .. [3] K. N. Abazajian et al. 2009, ApJS, 182, 543
#
