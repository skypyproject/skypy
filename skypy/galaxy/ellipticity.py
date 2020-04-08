"""Galaxy ellipticity module.

This module provides facilities to sample the ellipticities of galaxies.
"""

from scipy import stats


def _kacprzak_parameterization(beta_method):
    def reparameterize(self, *args):
        *beta_args, e_ratio, e_sum = args
        a = e_sum * e_ratio
        b = e_sum * (1.0 - e_ratio)
        return beta_method(self, *beta_args, a, b)
    return reparameterize


class kacprzak_gen(stats._continuous_distns.beta_gen):
    r'''Galaxy ellipticities sampled from a reparameterized beta distribution.

    The ellipticities follow a beta distribution parameterized by
    :math:`e_{\rm ratio}` and :math:`e_{\rm sum}` as presented in Kacprzak et
    al. 2019.

    Parameters
    ----------
    e_ratio : array_like
        Mean ellipticity of the distribution, must be between 0 and 1.
    e_sum : array_like
        Parameter controlling the width of the distribution, must be positive.

    Notes
    -----
    The probability distribution function :math:`p(e)` for ellipticity
    :math:`e` is given by a beta distribution:

    .. math::

        p(e) \sim \frac{\Gamma(a+b)}{\Gamma(a) \Gamma(b)} x^{a-1} (1-x)^{b-1}

    for :math:`0 <= e <= 1`, :math:`a = e_{\rm sum} e_{\rm ratio}`,
    :math:`b = e_{\rm sum} (1 - e_{\rm ratio})`, :math:`0 < e_{\rm ratio} < 1`
    and :math:`e_{\rm sum} > 0`, where :math:`\Gamma` is the gamma function.

    References
    ----------
    [1] Kacprzak T., Herbel J., Nicola A. et al., arXiv:1906.01018

    Examples
    --------
    >>> from skypy.galaxy.ellipticity import kacprzak

    Sample 10 random variates from the Kacprzak model with
    :math:`e_{\rm ratio} = 0.5` and :math:`e_{\rm sum} = 1.0`:

    >>> ellipticity = kacprzak.rvs(0.5, 1.0, size=10)

    Fix distribution parameters for repeated use:

    >>> kacprzak_distribution = kacprzak(0.5, 1.0)
    >>> kacprzak_distribution.mean()
    0.5
    >>> ellipticity = kacprzak_distribution.rvs(size=10)
    '''

    @_kacprzak_parameterization
    def _rvs(self, *args):
        return super()._rvs(*args)

    @_kacprzak_parameterization
    def _logpdf(self, *args):
        return super()._logpdf(*args)

    @_kacprzak_parameterization
    def _cdf(self, *args):
        return super()._cdf(*args)

    @_kacprzak_parameterization
    def _ppf(self, *args):
        return super()._ppf(*args)

    @_kacprzak_parameterization
    def _stats(self, *args):
        return super()._stats(*args)

    def fit(self, data, *args, **kwds):
        return super(stats._continuous_distns.beta_gen, self).fit(
            data, *args, **kwds)


kacprzak = kacprzak_gen(a=0.0, b=1.0, name='kacprzak', shapes="e_ratio e_sum")
