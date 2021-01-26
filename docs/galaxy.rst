*************************
Galaxies (`skypy.galaxy`)
*************************


Introduction
============


Galaxy Properties
=================

What follows are the physical properties of galaxies simulated by SkyPy, and
the available models for each individual property.


Luminosity
----------

The following models are found in the `skypy.galaxy.luminosity` package.

.. currentmodule:: skypy.galaxy.luminosity
.. autosummary::
   :nosignatures:

   schechter_lf_magnitude


Morphology
----------

The following models are found in the `skypy.galaxy.morphology` package.

.. currentmodule:: skypy.galaxy.morphology
.. autosummary::
   :nosignatures:

   angular_size
   beta_ellipticity
   early_type_lognormal_size
   late_type_lognormal_size
   linear_lognormal_size
   ryden04_ellipticity


Redshift
--------

The following models are found in the `skypy.galaxy.redshift` package.

.. currentmodule:: skypy.galaxy.redshift
.. autosummary::
   :nosignatures:

   redshifts_from_comoving_density
   schechter_lf_redshift
   schechter_smf_redshift
   smail


Spectrum
--------

The following models are found in the `skypy.galaxy.spectrum` package.

.. currentmodule:: skypy.galaxy.spectrum
.. autosummary::
   :nosignatures:

   dirichlet_coefficients
   mag_ab
   SpectrumTemplates
   KCorrectTemplates
   kcorrect


Stellar mass
------------

The following models are found in the `skypy.galaxy.stellar_mass` package.

.. currentmodule:: skypy.galaxy.stellar_mass
.. autosummary::
  :nosignatures:

  schechter_smf_mass


Reference/API
=============

.. automodapi:: skypy.galaxy
.. automodapi:: skypy.galaxy.luminosity
   :include-all-objects:
.. automodapi:: skypy.galaxy.morphology
.. automodapi:: skypy.galaxy.redshift
   :include-all-objects:
.. automodapi:: skypy.galaxy.spectrum
   :include-all-objects:
.. automodapi:: skypy.galaxy.stellar_mass
