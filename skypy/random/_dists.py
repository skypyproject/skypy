import numpy as np
from scipy import stats
import scipy.special as sc


def _gammainc(a, x):
    '''gammainc for negative indices'''
    b = a
    while b < 0:
        b = b+1
    g, f = sc.gammainc(b, x), sc.gammainc(b+1, x)
    while b > a:
        b, g, f = b-1, g+b/x*(g-f), g
    return g

gammainc = np.vectorize(_gammainc)

def _gammaincc(a, x):
    '''gammaincc for negative indices'''
    b = a
    while b < 0:
        b = b+1
    g, f = sc.gammaincc(b, x), sc.gammaincc(b+1, x)
    while b > a:
        b, g, f = b-1, g+b/x*(g-f), g
    return g

gammaincc = np.vectorize(_gammaincc)


class genschechter_gen(stats.rv_continuous):
    r'''Generalised Schechter random variable.

    The generalised Schechter distribution is a generalised gamma distribution
    with negative shape parameter :math:`\alpha` and a lower cut-off.

    Parameters
    ----------
    alpha : float or array_like of floats
        Power law exponent :math:`x^\alpha`, must greater than -2.
    gamma : float or array_like of floats
        Log-power law exponent :math:`\exp\{-x^\gamma\}`, must be positive.
    a : float or array_like of floats
        Lower cut-off of the distribution.

    Notes
    -----
    The probability distribution function for `genschechter` is

    .. math::

        f(x) = \frac{\gamma \, x^\alpha \, e^{-x^\gamma}}
                {\Gamma\Bigl(\frac{\alpha+1}{\gamma}, c^\gamma\Bigr)} \;,

    for :math:`x \ge a`, where :math:`a` is given as a shape parameter, and
    :math:`\alpha`, :math:`\gamma` are the shape parameters of the distribution.

    Examples
    --------
    >>> from skypy.random import genschechter

    Sample 10 random variates from the `genschechter` distribution with negative
    shape parameter `alpha = -1.2` and lower limit `a = 1e-5`.

    >>> x = genschechter.rvs(-1.2, 1, 1e-5, size=10)
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
        Gn = np.log(np.fabs(gammaincc(ap1nog, a_to_gamma))) + sc.gammaln(ap1nog)
        G = np.log(np.fabs(gammaincc(ap1og, a_to_gamma))) + sc.gammaln(ap1og)
        return np.exp(Gn - G)

genschechter = genschechter_gen(a=0., name='genschechter')
