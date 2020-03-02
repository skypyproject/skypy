"""Galaxy morphology module.

This module provides facilities to sample the morphological properties of
galaxies using a number of models.
"""

from scipy.stats._continuous_distns import beta_gen


def _kacprzak_parameterization(beta_method):
    def reparameterize(self, *args):
        *beta_args, e_sum, e_ratio = args
        a = e_sum * e_ratio
        b = e_sum * (1.0 - e_ratio)
        return beta_method(self, *beta_args, a, b)
    return reparameterize


class ellipticity_beta_gen(beta_gen):
    r'''Galaxy ellipticities sampled from a reparameterized beta distribution.

    The ellipticities follow a beta distribution parameterized by
    :math:`e_{\rm sum}` and :math:`e_{\rm ratio}` as presented in Kacprzak et
    al. 2019.

    Parameters
    ----------
    e_sum : array_like
        Parameter controlling the width of the distribution, must be positive.
    e_ratio : array_like
        Mean ellipticity of the distribution, must be between 0 and 1.

    Notes
    -----
    The probability distribution function :math:`p(e)` for ellipticity :math:`e`
    is given by a beta distribution:

    .. math::

        p(e) \sim \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha) \Gamma(\beta)}
                  x^{\alpha-1} (1-x)^{\beta-1}

    for :math:`0 <= e <= 1`, :math:`\alpha = e_{\rm sum} e_{\rm ratio}`,
    :math:`\beta = e_{\rm sum} (1 - e_{\rm ratio})`, :math:`e_{\rm sum} > 0`
    and :math:`0 < e_{\rm ratio} < 1`, where :math:`\Gamma` is the gamma
    function.

    References
    ----------
    [1] Kacprzak T., Herbel J., Nicola A. et al., arXiv:1906.01018

    Examples
    --------
    >>> from skypy.galaxy.morphology import ellipticity_beta

    Sample 10 random variates from the Kacprzak model with
    :math:`e_{\rm sum} = 1.0` and :math:`e_{\rm ratio} = 0.5`:

    >>> ellipticity = ellipticity_beta.rvs(1.0, 0.5, size=10)

    Fix distribution parameters for repeated use:

    >>> ellipticity_dist = ellipticity_beta(1.0, 0.5)
    >>> ellipticity_dist.mean()
    0.5
    >>> ellipticity = ellipticity_dist.rvs(size=10)
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
        return super(beta_gen, self).fit(data, *args, **kwds)


ellipticity_beta = ellipticity_beta_gen(a=0.0, b=1.0, name='ellipticity_beta')
