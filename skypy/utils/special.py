"""Special functions.

This module computes useful special functions.

.. autosummary::
   :nosignatures:
   :toctree: ../api/

   gammaincc

"""

import numpy as np
import scipy.special as sc


def _gammaincc(a, x):
    '''gammaincc for positive or negative indices and scalar inputs'''
    if x < 0:
        raise ValueError('negative x in gammaincc')
    if x == 0:
        if a > 0:
            return 1
        s = sc.gammasgn(a)
        if s == 0:
            return 0
        return s*np.inf
    if np.isinf(x):
        return 0
    if a < 0:
        n, b = np.divmod(a, 1)
    else:
        n, b = 0, a
    g = sc.gammaincc(b, x)
    if n < 0:
        f = np.exp(sc.xlogy(b, x) - x - sc.gammaln(b+1))
        while n < 0:
            f *= (a-n)/x
            g -= f
            n += 1
    return g


def gammaincc(a, x):
    '''Regularized upper incomplete gamma function.

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
    `scipy.special.gammaincc` for arguments :math:`b, x` where :math:`b` is the
    lowest nonnegative number such that :math:`b = a + n`, :math:`n` integer.

    See also
    --------
    scipy.special.gammaincc : Computes the start of the recurrence.

    Examples
    --------
    This implementation of `gammaincc` supports positive and negative values
    of `a`.

    >>> from skypy.utils.special import gammaincc
    >>> gammaincc([-1.5, -0.5, 0.5, 1.5], 1.2)
    array([ 0.03084582, -0.03378949,  0.12133525,  0.49363462])

    '''
    if np.broadcast(a, x).ndim == 0:
        return _gammaincc(a, x)
    a, x = np.broadcast_arrays(a, x)
    if np.any(x < 0):
        raise ValueError('negative x in gammaincc')

    # nonpositive a need special treatment
    i = a <= 0

    # find n, b such that a + n = b with b >= 0 and n integer
    n = np.zeros_like(a)
    b = np.copy(a)
    np.divmod(b, 1, out=(n, b), where=i)

    # compute gammaincc for b and x as usual
    g = sc.gammaincc(b, x)

    # deal with nonpositive a
    # the number n keeps track of iterations still to do
    if np.sum(i) > 0:
        # all x = inf are done
        n[i & np.isinf(x)] = 0

        # do x = 0 for nonpositive a; depends on Gamma(a)
        i = i & (x == 0)
        s = sc.gammasgn(a[i])
        g[i] = np.where(s == 0, 0, s*np.inf)
        n[i] = 0

        # these are still to do
        i = n < 0

        # recurrence
        f = np.empty_like(g)
        f[i] = np.exp(sc.xlogy(b[i], x[i]) - x[i] - sc.gammaln(b[i]+1))
        while np.sum(i) > 0:
            f[i] *= (a[i]-n[i])/x[i]
            g[i] -= f[i]
            n[i] += 1
            i[i] = n[i] < 0
    return g
