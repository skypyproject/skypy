# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Skypy is a package offering core functionality and common tools for
astronomical forward-modelling in Python. It contains methods for modelling
the Universe, galaxies and the Milky Way and for generating synthetic
observational data.
"""

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------

__all__ = []

from . import galaxies  # noqa
from . import pipeline  # noqa
