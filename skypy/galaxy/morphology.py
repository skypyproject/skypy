"""Galaxy morphology module.

This module provides facilities to sample the morphological properties of
galaxies using a number of models.
"""

from scipy import stats


def _kacprzak_input(beta_method):
    def reparameterize(self, *args):
        *beta_args, e_sum, e_ratio = args
        a = e_sum * e_ratio
        b = e_sum * (1.0 - e_ratio)
        return beta_method(self, *beta_args, a, b)
    return reparameterize


def _kacprzak_output(beta_method):
    def reparameterize(self, *args, **kwds):
        a, b, *beta_out = beta_method(self, *args, **kwds)
        e_sum = a + b
        e_ratio = a / (a + b)
        return (e_sum, e_ratio, *beta_out)
    return reparameterize


class ellipticity_beta_gen(stats._continuous_distns.beta_gen):

    @_kacprzak_input
    def _rvs(self, *args):
        return super()._rvs(*args)

    @_kacprzak_input
    def _logpdf(self, *args):
        return super()._logpdf(*args)

    @_kacprzak_input
    def _cdf(self, *args):
        return super()._cdf(*args)

    @_kacprzak_input
    def _ppf(self, *args):
        return super()._ppf(*args)

    @_kacprzak_input
    def _stats(self, *args):
        return super()._stats(*args)

    @_kacprzak_output
    def fit(self, data, *args, **kwds):
        return super().fit(data, *args, **kwds)


ellipticity_beta = ellipticity_beta_gen(a=0.0, b=1.0, name='ellipticity_beta')
