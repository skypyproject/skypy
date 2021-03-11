"""Halos module.

This module contains methods that model the properties of dark matter halo
populations.

Models
======
.. autosummary::
   :nosignatures:
   :toctree: ../api/

   colossus_mf
"""

__all__ = [
    'colossus_mf',
]

from . import abundance_matching  # noqa F401,F403
from . import mass  # noqa F401,F403
from . import quenching  # noqa F401,F403
from ._colossus import colossus_mf  # noqa F401,F403
