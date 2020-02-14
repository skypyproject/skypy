"""Galaxy redshift module.

This module provides facilities to sample galaxy redshifts using a number of
models.
"""

import numpy as np
from scipy import stats
import scipy.special as sc


class smail_gen(stats.rv_continuous):
    r'''Redshifts following the Smail et al. (1994) model.

    The redshift follows the Smail et al. (1994) redshift distribution as
    reported by Amara & Refregier (2007).

    Parameters
    ----------
    z_median : float or array_like of floats
        Median redshift of the distribution, must be positive.
    alpha : float or array_like of floats
        Power law exponent (z/z0)^\alpha, must be positive.
    beta : float or array_like of floats
        Log-power law exponent exp[-(z/z0)^\beta], must be positive.

    Notes
    -----
    The probability distribution function :math:`p(z)` for redshift :math:`z`
    is given by Amara & Refregier (2007) as

    .. math::

        p(z) \sim \left(\frac{z}{z_0}\right)^\alpha
                    \exp\left[-\left(\frac{z}{z_0}\right)^\beta\right] \;.

    This is brought into the form of a gamma distribution by a change of
    variable :math:`z \to x = z^\beta`.

    References
    ----------
    [1] Smail I., Ellis R. S., Fitchett M. J., 1994, MNRAS, 270, 245
    [2] Amara A., Refregier A., 2007, MNRAS, 381, 1018

    Examples
    --------
    >>> from skypy.galaxy.redshift import smail

    Sample 10 random variates from the Smail model with `alpha = 1.5` and
    `beta = 2` and median redshift `z_median = 1.2`.

    >>> redshift = smail.rvs(1.2, 1.5, 2.0, size=10)

    Fix distribution parameters for repeated use.

    >>> redshift_dist = smail(1.2, 1.5, 2.0)
    >>> redshift_dist.median()
    1.2
    >>> redshift = redshift_dist.rvs(size=10)
    '''

    def _rvs(self, zm, a, b):
        sz, rs = self._size, self._random_state
        k = (a+1)/b
        t = zm**b/sc.gammainccinv(k, 0.5)
        g = stats.gamma.rvs(k, scale=t, size=sz, random_state=rs)
        return g**(1/b)

    def _pdf(self, z, zm, a, b):
        return np.exp(self._logpdf(z, zm, a, b))

    def _logpdf(self, z, zm, a, b):
        k = (a+1)/b
        z0 = zm/sc.gammainccinv(k, 0.5)**(1/b)
        lognorm = np.log(b) - np.log(z0) - sc.gammaln(k)
        return lognorm + sc.xlogy(a, z/z0) - (z/z0)**b

    def _cdf(self, z, zm, a, b):
        k = (a+1)/b
        t = sc.gammainccinv(k, 0.5)
        return sc.gammainc(k, t*(z/zm)**b)

    def _ppf(self, q, zm, a, b):
        k = (a+1)/b
        t = sc.gammainccinv(k, 0.5)
        return zm*(sc.gammaincinv(k, q)/t)**(1/b)

    def _sf(self, z, zm, a, b):
        k = (a+1)/b
        t = sc.gammainccinv(k, 0.5)
        return sc.gammaincc(k, t*(z/zm)**b)

    def _isf(self, q, zm, a, b):
        k = (a+1)/b
        t = sc.gammainccinv(k, 0.5)
        return zm*(sc.gammainccinv(k, q)/t)**(1/b)

    def _munp(self, n, zm, a, b):
        k = (a+1)/b
        z0 = zm/sc.gammainccinv(k, 0.5)**(1/b)
        return z0**n*np.exp(sc.gammaln((a+n+1)/b) - sc.gammaln(k))


smail = smail_gen(a=0., name='smail', shapes='z_median, alpha, beta')
