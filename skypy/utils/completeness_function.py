"""Completeness functions.

This module provides a set of completeness functions for different cosmological surveys.

"""

import numpy as np

__all__ = [
    'logistic_completeness_function',
]

def logistic_completeness_function(magnitude, magnitude_95, magnitude_50):
    r'''Compute logistic completeness function.

    This function calculates the logistic completeness function (based on eq. (7) in [1]_)

    .. math::

        p(m) = \frac{1}{1 + \exp[\kappa (m - m_{50})]}\;,

    which describes the probability :math:`p(m)` that an object of magnitude :math:`m` is detected
    in the band and with :math:`\kappa = \frac{\ln(1/19)}{m_{95} - m_{50}}`.

    Parameters
    ----------
    magnitude : array_like
        Magnitudes. Can be multidimensional for computing with multiple filter bands.
    magnitude_95 : scalar or 1-D array_like
        95% completeness magnitude.
        If magnitude_50 is 1-D array it has to be scalar or 1-D array of the same shape.
    magnitude_50 : scalar or 1-D array_like
        50% completeness magnitude.
        If magnitude_95 is 1-D array it has to be scalar or 1-D array of the same shape.

    Returns
    -------
    probability : scalar or array_like
        Probability of detecting an object with magnitude :math:`m`.
        Returns array_like of the same shape as magnitude.
        Exemption: If magnitude is scalar and magnitude_95 or magnitude_50
        is array_like of shape (nb, ) it returns array_like of shape (nb, ).

    References
    -----------
    ..[1] Lopez-Sanjuan C. et al., 2017, A&A, 599, A62

    '''

    kappa = np.log(1. / 19) / np.subtract(magnitude_95, magnitude_50)
    return 1. / (1 + np.exp(kappa * np.subtract(magnitude, magnitude_50)))
