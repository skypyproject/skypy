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


# absolute AB magnitude in various bands in terms of solar luminosity
# values depend on the particular bandpass used and are approximate
standard_bandpass_zeropoints = {
    'U': -4.98,
    'B': -4.73,
    'V': -4.48,
}


def luminosity_from_absolute_magnitude(absolute_magnitude, band=None):
    """ Converts absolute magnitudes into luminosities

    Parameters
    ----------
    absolute_magnitude : array_like
        Input absolute magnitudes
    band : str, optional
        Standard bandpass for conversion.

    Returns
    -------
    ndarray, or float if input is scalar
    Luminosity values. If a standard bandpass is given, the luminosity is in
    units of solar luminosity.
    """

    zeropoint = 0
    if band is not None:
        zeropoint = standard_bandpass_zeropoints[band]

    return 10.**(-0.4*np.add(absolute_magnitude, zeropoint))


def absolute_magnitude_from_luminosity(luminosity, band=None):
    """ Converts luminosities into absolute magnitudes

    Parameters
    ----------
    luminosity : array_like
        Input luminosity
    band : str, optional
        Standard bandpass for conversion.

    Returns
    -------
    ndarray, or float if input is scalar
    Absolute magnitude values. If a standard bandpass is given, the luminosity
    is assumed in units of solar luminosity.
    """

    zeropoint = 0
    if band is not None:
        zeropoint = standard_bandpass_zeropoints[band]

    return -2.5*np.log10(luminosity) - zeropoint
