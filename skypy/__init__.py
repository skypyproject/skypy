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
# from .example_mod import *   # noqa
# Then you can be explicit to control what ends up in the namespace,
#__all__ += ['do_primes']   # noqa
# or you can keep everything from the subpackage with the following instead
# __all__ += example_mod.__all__

from . import cluster  # noqa
from . import galaxy  # noqa
from . import gravitational_wave  # noqa
from . import halo  # noqa
from . import power_spectrum  # noqa
from . import supernova  # noqa
