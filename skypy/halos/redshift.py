"""Halo redshifts.

This module implements functions to sample from redshift distributions of dark
matter halos.

Models
======
.. autosummary::
   :nosignatures:
   :toctree: ../api/

   colossus_mf_redshift
"""

from ._colossus import colossus_mf_redshift

__all__ = [
    'colossus_mf_redshift',
]
