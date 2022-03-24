***************************
Galaxies (`skypy.galaxies`)
***************************


Introduction
============


Galaxy Properties
=================

What follows are the physical properties of galaxies simulated by SkyPy, and
the available models for each individual property.


Luminosity
----------

The following models are found in the `skypy.galaxies.luminosity` package.

.. currentmodule:: skypy.galaxies.luminosity
.. autosummary::
   :nosignatures:

   schechter_lf_magnitude


Morphology
----------

The following models are found in the `skypy.galaxies.morphology` package.

.. currentmodule:: skypy.galaxies.morphology
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

The following models are found in the `skypy.galaxies.redshift` package.

.. currentmodule:: skypy.galaxies.redshift
.. autosummary::
   :nosignatures:

   redshifts_from_comoving_density
   schechter_lf_redshift
   schechter_smf_redshift
   smail


Spectrum
--------

The following models are found in the `skypy.galaxies.spectrum` package.

.. currentmodule:: skypy.galaxies.spectrum
.. autosummary::
   :nosignatures:

   dirichlet_coefficients
   KCorrectTemplates
   kcorrect


Stellar mass
------------

The following models are found in the `skypy.galaxies.stellar_mass` package.

.. currentmodule:: skypy.galaxies.stellar_mass
.. autosummary::
  :nosignatures:

  schechter_smf_mass


Reference/API
=============

.. automodapi:: skypy.galaxies
.. automodapi:: skypy.galaxies.luminosity
   :include-all-objects:
.. automodapi:: skypy.galaxies.morphology
.. automodapi:: skypy.galaxies.redshift
   :include-all-objects:
.. automodapi:: skypy.galaxies.spectrum
   :include-all-objects:
.. automodapi:: skypy.galaxies.stellar_mass
