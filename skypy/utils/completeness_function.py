"""Completeness functions.

This module computes completeness functions of cosmological surveys.

.. autosummary::
   :nosignatures:
   :toctree: ../api/

   logistic_completeness_function

"""

import numpy as np

#__all__ = [
 #   'logistic_completeness_function',
#]

def logistic_completeness_function(m, m95, m50):
    r'''Compute logistic completeness function.

    This function calculates the logistic completeness function

    .. math::

        p(m) = \frac{1}{1 + \exp\left[\frac{\ln(1/19)}{m_{95} - m_{50}} (m - m_{50})\right]}\;.

    which describes the probability :math:`p(m)` that an object of magnitude :math:`m` is detected in
    the band.

    Parameters
    ----------
    m : array_like
        Magnitudes. Can be multidimensional for computing with multiple filter bands.
    m95 : array_like
        95% completeness magnitude.
    m50 : array_like
        50% completeness magnitude.
    Returns
    -------
    p : array like
        Result of the completeness function.

    '''

    return 1. / (1 + np.exp(np.log(1. / 19) / np.subtract(m95, m50) * np.subtract(m, m50)))