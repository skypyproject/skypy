"""Astronomy utility module.

This module provides methods to convert among astronomical quantities
like luminosity and magnitude.

Utility functions
=================

.. autosummary::
   :nosignatures:
   :toctree: ../api/

   luminosity_from_absolute_magnitude
   absolute_magnitude_from_luminosity

"""

import numpy as np


def luminosity_from_absolute_magnitude(absolute_magnitude, zeropoint=None):
    """ Converts absolute magnitudes into luminosities

    Parameters
    ----------
    absolute_magnitude : array_like
        Input absolute magnitudes
    zeropoint : float, optional
        Zeropoint for the conversion.

    Returns
    -------
    ndarray, or float if input is scalar
    Luminosity values.
    """

    if zeropoint is None:
        zeropoint = 0.

    return 10.**(-0.4*np.add(absolute_magnitude, zeropoint))


def absolute_magnitude_from_luminosity(luminosity, zeropoint=None):
    """ Converts luminosities into absolute magnitudes

    Parameters
    ----------
    luminosity : array_like
        Input luminosity
    zeropoint : float, optional
        Zeropoint for the conversion.

    Returns
    -------
    ndarray, or float if input is scalar
    Absolute magnitude values.
    """

    if zeropoint is None:
        zeropoint = 0.

    return -2.5*np.log10(luminosity) - zeropoint
