"""Special functions.

This module computes useful special functions.

.. autosummary::
   :nosignatures:
   :toctree: ../api/

   gammaincc

"""

from importlib.metadata import version
from packaging.version import parse

import numpy as np
import scipy.special as sc


def _is_gamma_pole(a):
    '''Utility function that returns True if a is a pole of the gamma function'''
    s = sc.gammasgn(a)
    if parse(version('scipy')) < parse('1.15'):
        return s == 0
    else:
        return np.logical_or(a == 0, np.isnan(s))


def _gammaincc(a, x):
    '''gammaincc for positive or negative indices and scalar inputs'''
    if x < 0:
        raise ValueError('negative x in gammaincc')
    if x == 0:
        if a > 0:
            return 1
        if _is_gamma_pole(a):
            return 0
        return sc.gammasgn(a) * np.inf
    if np.isinf(x):
        return 0
    if a < 0:
        n = np.floor(a)
    else:
        n = 0
    g = sc.gammaincc(a-n, x) if a != n else 0
    if n < 0:
        f = np.exp(sc.xlogy(a-n, x)-x-sc.gammaln(a-n+1))
        while n < 0:
            f *= (a-n)/x
            g -= f
            n += 1
    return g


def gammaincc(a, x):
    r'''Regularized upper incomplete gamma function.

    This implementation of `gammaincc` allows :math:`a` real and :math:`x`
    nonnegative.

    Parameters
    ----------
    a : array_like
        Real parameter.

    x : array_like
        Nonnegative argument.

    Returns
    -------
    scalar or ndarray
        Values of the upper incomplete gamma function.

    Notes
    -----
    The function value is computed via a recurrence from the value of
    `scipy.special.gammaincc` for arguments :math:`a-n, x` where :math:`n` is
    the smallest integer such that :math:`a-n \ge 0`.

    See also
    --------
    scipy.special.gammaincc : Computes the start of the recurrence.

    '''
    if np.broadcast(a, x).ndim == 0:
        return _gammaincc(a, x)
    a, x = np.broadcast_arrays(a, x)
    if np.any(x < 0):
        raise ValueError('negative x in gammaincc')

    # nonpositive a need special treatment
    i = a <= 0

    # find integer n such that a + n >= 0
    n = np.where(i, np.floor(a), 0)

    # compute gammaincc for a-n and x as usual
    g = np.where(a == n, 0, sc.gammaincc(a-n, x))

    # deal with nonpositive a
    # the number n keeps track of iterations still to do
    if np.any(i):
        # all x = inf are done
        n[i & np.isinf(x)] = 0

        # do x = 0 for nonpositive a; depends on Gamma(a)
        i = i & (x == 0)
        s = sc.gammasgn(a[i])
        g[i] = np.where(_is_gamma_pole(a[i]), 0, np.copysign(np.inf, s))
        n[i] = 0

        # these are still to do
        i = n < 0

        # recurrence
        f = np.empty_like(g)
        f[i] = np.exp(sc.xlogy(a[i]-n[i], x[i])-x[i]-sc.gammaln(a[i]-n[i]+1))
        while np.any(i):
            f[i] *= (a[i]-n[i])/x[i]
            g[i] -= f[i]
            n[i] += 1
            i[i] = n[i] < 0
    return g
