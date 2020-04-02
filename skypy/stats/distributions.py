import numpy as np
from ._dist_infrastructure import rv_continuous
import scipy.special as sc


def gammaincc_m1(a, x):
    '''gammaincc for negative indices > -1'''
    return sc.gammaincc(a+1, x) - np.exp(sc.xlogy(a, x) - x - sc.gammaln(a+1))


def _gammaincc(a, x):
    '''gammaincc for negative indices'''
    b = a
    while b < 0:
        b = b+1
    g = sc.gammaincc(b, x)
    f = np.exp(sc.xlogy(b, x) - x - sc.gammaln(b+1))
    while b > a:
        f *= b/x
        g -= f
        b -= 1
    return g


gammaincc = np.vectorize(_gammaincc, otypes=['float'])


class schechter_gen(rv_continuous):
    r'''Schechter random variable.

    The Schechter distribution is a gamma distribution with negative shape
    parameter :math:`\alpha` and a lower cut-off.

    %(before_notes)s

    Notes
    -----
    The probability distribution function for `schechter` is

    .. math::

        f(x) = \frac{x^\alpha \, e^{-x}}{\Gamma(\alpha+1, a)} \;, \quad
        x \ge a \;,

    where :math:`\alpha > -2` is the shape parameter of the distribution, and
    :math:`a > 0` is the shape parameter for the lower cut-off of the support.

    %(after_notes)s

    %(example)s

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

    def _munp(self, n, alpha, a):
        ap1n = alpha+1+n
        ap1 = alpha+1
        u = np.log(np.fabs(gammaincc_m1(ap1n, a))) + sc.gammaln(ap1n)
        v = np.log(np.fabs(gammaincc_m1(ap1, a))) + sc.gammaln(ap1)
        return np.exp(u - v)


schechter = schechter_gen(a=0., name='schechter')


class genschechter_gen(rv_continuous):
    r'''Generalised Schechter random variable.

    The generalised Schechter distribution is a generalised gamma distribution
    with negative shape parameter :math:`\alpha` and a lower cut-off.

    %(before_notes)s

    Notes
    -----
    The probability distribution function for `genschechter` is

    .. math::

        f(x) = \frac{\gamma \, x^\alpha \, e^{-x^\gamma}}
                {\Gamma\Bigl(\frac{\alpha+1}{\gamma}, a^\gamma\Bigr)} \;,

    for :math:`x \ge a`, where :math:`\alpha > -2` and :math:`\gamma > 0` are
    the shape parameters of the distribution, and:math:`a > 0` is the shape
    parameter for the lower cut-off of the support.

    %(after_notes)s

    %(example)s

    '''

    def _argcheck(self, alpha, gamma, a):
        return (alpha > -2) & (gamma > 0) & (a > 0)

    def _get_support(self, alpha, gamma, a):
        return a, self.b

    def _pdf(self, x, alpha, gamma, a):
        return np.exp(self._logpdf(x, alpha, gamma, a))

    def _logpdf(self, x, alpha, gamma, a):
        ap1og = (alpha+1)/gamma
        norm = np.log(np.fabs(gammaincc(ap1og, a**gamma))) + sc.gammaln(ap1og)
        return sc.xlogy(alpha, x) - x**gamma + np.log(gamma) - norm

    def _cdf(self, x, alpha, gamma, a):
        return 1 - self._sf(x, alpha, gamma, a)

    def _sf(self, x, alpha, gamma, a):
        ap1og = (alpha+1)/gamma
        return gammaincc(ap1og, x**gamma)/gammaincc(ap1og, a**gamma)

    def _munp(self, n, alpha, gamma, a):
        a_to_gamma = a**gamma
        ap1nog = (alpha+1+n)/gamma
        ap1og = (alpha+1)/gamma
        u = np.log(np.fabs(gammaincc(ap1nog, a_to_gamma))) + sc.gammaln(ap1nog)
        v = np.log(np.fabs(gammaincc(ap1og, a_to_gamma))) + sc.gammaln(ap1og)
        return np.exp(u - v)


genschechter = genschechter_gen(a=0., name='genschechter')
