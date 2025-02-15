# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ['__version__']

try:
    from .version import version as __version__
except ImportError:
    __version__ = ''
