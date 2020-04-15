import numpy as np
from scipy.stats import rv_continuous
from .rv import examples
import scipy.special as sc


# list of exported distributions
__all__ = [
    'schechter',
]


def gammaincc_m1(a, x):
    '''gammaincc for negative indices > -1'''
    return sc.gammaincc(a+1, x) - np.exp(sc.xlogy(a, x) - x - sc.gammaln(a+1))


def _gammaincinv_pm1_iter(a, t, sgx, glna, oma, tol):
    dt = np.inf
    while dt > tol:
        u = np.fabs(gammaincc_m1(a, t)) - sgx
        dt = u*np.exp(t + sc.xlogy(oma, t) + glna)
        t += dt
    return t


def gammainccinv_pm1(a, x, tol=1e-8):
    '''gammainccinv for indices -1 < a < 1

    Reference: Gil et al. (2012) for 0 < a < 1; NT
    '''
    sgx = np.sign(a)*x
    t = np.exp((np.log(1-x) + sc.gammaln(a+1))/a)
    glna = sc.gammaln(a)
    oma = 1-a
    return np.vectorize(_gammaincinv_pm1_iter)(a, t, sgx, glna, oma, tol)


@examples(name='schechter', args=(-1.2, 10.))
class schechter_gen(rv_continuous):
    r'''Schechter random variable.

    The Schechter distribution is a gamma distribution with negative shape
    parameter :math:`\alpha` and a lower cut-off.

    Parameters
    ----------
    alpha :
        The exponent :math:`0 > \alpha > -2` of the power law :math:`x^\alpha`.
    a :
        The left truncation of the distribution, :math:`x \ge a`.

    Notes
    -----
    The probability distribution function for `schechter` is

    .. math::

        f(x) = \frac{x^\alpha \, e^{-x}}{\Gamma(\alpha+1, a)} \;, \quad
        x \ge a \;,

    where :math:`\alpha > -2` is the shape parameter of the distribution, and
    :math:`a > 0` is the shape parameter for the lower cut-off of the support.

    '''

    def _argcheck(self, alpha, a):
        return (alpha > -2) & (a > 0)

    def _get_support(self, alpha, a):
        return a, self.b

    def _pdf(self, x, alpha, a):
        return np.exp(self._logpdf(x, alpha, a))

    def _logpdf(self, x, alpha, a):
        ap1 = alpha+1
        norm = np.log(np.fabs(gammaincc_m1(ap1, a))) + sc.gammaln(ap1)
        return sc.xlogy(alpha, x) - x - norm

    def _cdf(self, x, alpha, a):
        return 1 - self._sf(x, alpha, a)

    def _sf(self, x, alpha, a):
        ap1 = alpha+1
        return gammaincc_m1(ap1, x)/gammaincc_m1(ap1, a)

    def _ppf(self, q, alpha, a):
        return self._isf(1-q, alpha, a)

    def _isf(self, q, alpha, a):
        ap1 = alpha+1
        return gammainccinv_pm1(ap1, q*gammaincc_m1(ap1, a))

    def _munp(self, n, alpha, a):
        ap1n = alpha+1+n
        ap1 = alpha+1
        u = np.log(np.fabs(gammaincc_m1(ap1n, a))) + sc.gammaln(ap1n)
        v = np.log(np.fabs(gammaincc_m1(ap1, a))) + sc.gammaln(ap1)
        return np.exp(u - v)


schechter = schechter_gen(a=0., name='schechter')
