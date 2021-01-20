*************************
Galaxies (`skypy.galaxy`)
*************************


Introduction
============


Galaxy Properties
=================

What follows are the physical properties of galaxies simulated by SkyPy, and
the available models for each individual property.


Ellipticity
-----------

The following models are found in the `skypy.galaxy.ellipticity` package.

.. currentmodule:: skypy.galaxy.ellipticity
.. autosummary::
   :nosignatures:

   beta_ellipticity
   ryden04


Luminosity
----------

The following models are found in the `skypy.galaxy.luminosity` package.

.. currentmodule:: skypy.galaxy.luminosity
.. autosummary::
   :nosignatures:

   schechter_lf_magnitude


Redshift
--------

The following models are found in the `skypy.galaxy.redshift` package.

.. currentmodule:: skypy.galaxy.redshift
.. autosummary::
   :nosignatures:

   redshifts_from_comoving_density
   schechter_lf_redshift
   smail


Size
----

The following models are found in the `skypy.galaxy.size` package.

.. currentmodule:: skypy.galaxy.size
.. autosummary::
   :nosignatures:

   angular_size
   early_type_lognormal
   late_type_lognormal
   linear_lognormal


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

  schechter_smf


Reference/API
=============

.. automodapi:: skypy.galaxy
.. automodapi:: skypy.galaxy.luminosity
   :include-all-objects:
.. automodapi:: skypy.galaxy.ellipticity
   :include-all-objects:
.. automodapi:: skypy.galaxy.redshift
   :include-all-objects:
.. automodapi:: skypy.galaxy.size
.. automodapi:: skypy.galaxy.spectrum
.. automodapi:: skypy.galaxy.stellar_mass
