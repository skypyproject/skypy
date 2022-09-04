"""Sample from velocity distribution function

"""

import numpy as np
import scipy.special as special
from scipy.interpolate import interp1d

def sample_vdf(x_min, x_max, resolution=100, size=1):
    """Sample from velocity dispersion function of elliptical galaxies in the local universe [1]_.
    Parameters
    ----------
    xmin, xmax: int
        Lower and upper bounds of random variable x. Samples are drawn uniformly from bounds.
    resolution: int
        Resolution of the inverse transform sampling spline. Default is 100.
    size: int
        Number of samples returned. Default is 1.
    Returns
    -------
    x_sample: array_like
        Samples drawn from vdf function.
    Warnings
    --------
    Inverse cumulative dispersion function is approximated from the function 
    using quadratic interpolation. The usre should specify the resolution to 
    satisfy their numerical accuracy.
    References
    ----------
    .. [1] Choi, Park and Vogeley, (2007), astro-ph/0611607, doi:10.1086/511060
    """
    x = np.linspace(0, 1000, resolution)
    vdf_func = lambda x: 8e-3*(x/161)**2.32*np.exp(-(x/161)**2.67)*(2.67/special.gamma(2.32/2.67))*(1/x)
    y = vdf_func(x)
    pdf_sampler = PDFSampling(x, y[1:], x_min, x_max)
    return pdf_sampler.draw(size)


class PDFSampling(object):
    """
    class for approximations with a given pdf sample
    """
    def __init__(self, bin_edges, pdf_array, x_min, x_max):
        """
        :param bin_edges: bin edges of PDF values
        :param pdf_array: pdf array of given bins (len(bin_edges)-1)
        """
        assert len(bin_edges) == len(pdf_array) + 1
        self.x_min, self.x_max = x_min, x_max
        self._cdf_array, self._cdf_func, self._cdf_inv_func = approx_cdf_1d(bin_edges, pdf_array)

    def draw(self, n=1):
        """
        :return:
        """
        t_min, t_max = self._cdf_func(self.x_min), self._cdf_func(self.x_max)
        p = np.random.uniform(t_min, t_max, n)
        return self._cdf_inv_func(p)

    @property
    def draw_one(self):
        """
        :return:
        """
        return self.draw(n=1)


def approx_cdf_1d(bin_edges, pdf_array):

    """
    :param bin_edges: bin edges of PDF values
    :param pdf_array: pdf array of given bins (len(bin_edges)-1)
    :return: cdf, interp1d function of cdf, inverse interpolation function
    """
    assert len(bin_edges) == len(pdf_array) + 1
    norm_pdf = pdf_array/np.sum(pdf_array)
    cdf_array = np.zeros_like(bin_edges)
    cdf_array[0] = 0
    for i in range(0, len(norm_pdf)):
        cdf_array[i+1] = cdf_array[i] + norm_pdf[i]
    cdf_func = interp1d(bin_edges, cdf_array)
    cdf_inv_func = interp1d(cdf_array, bin_edges)
    return cdf_array, cdf_func, cdf_inv_func