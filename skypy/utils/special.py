"""Special functions.

This module computes useful special functions.


Utility functions
=================

.. autosummary::
   :nosignatures:
   :toctree: ../api/

   upper_incomplete_gamma

"""

import numpy as np

import scipy.special


def upper_incomplete_gamma(a, x):
    """ Non-regularised upper incomplete gamma function. Extension of the
    regularised upper incomplete gamma function implemented in SciPy. In
    this way you can pass a negative value for a.

    Parameters
    ----------
    a : array_like
        Parameter
    x : array_like
        Nonnegative parameter

    Returns
    -------
    Scalar or ndarray
        Value of the non-regularised upper incomplete gamma function.
    """
    if a > 0:
        return scipy.special.gammaincc(a, x) * scipy.special.gamma(a)
    return (scipy.special.gammaincc(a + 1, x)
            - np.power(x, a) * np.exp(-x) / scipy.special.gamma(a + 1)) \
        * scipy.special.gamma(a)
