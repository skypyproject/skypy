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
