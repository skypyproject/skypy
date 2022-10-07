r"""Models of galaxy velocity dispersion.

"""

import numpy as np
import scipy.special as special

__all__ = [
    'schechter_vdf',
]

def schecter_vdf(vd_min, vd_max, resolution=100, size=None, scale=1.):
    r"""Sample velocity dispersion of elliptical galaxies in the local universe following a Schecter function.

    Parameters
    ----------
    vd_min, vd_max: int
        Lower and upper bounds of random variable x. Samples are drawn uniformly from bounds.
    resolution: int
        Resolution of the inverse transform sampling spline. Default is 100.
    size: int
        Number of samples returned. Default is 1.

    Returns
    -------
    velocity_dispersion: array_like
        Velocity dispersion drawn from Schechter function.

    Notes
    -----
    The probability distribution function :math:`p(\sigma)` for velocity dispersion :math:`\sigma`
    can be described by a Schechter function (see eq. (4) in [1]_)

    .. math::
    
        \begin{eqnarray}&&{dn}={\phi }_{*}{(\displaystyle \frac{\sigma }{{\sigma }_{*}})}^{\alpha }
        \mathrm{exp}[-{(\displaystyle \frac{\sigma  }{{\sigma }_{*}})}^{\beta }]\displaystyle
        \frac{\beta }{{\rm{\Gamma }}(\alpha /\beta )}\displaystyle \frac{1}{\sigma },
        \end{eqnarray}\tag{3} \;.

    References
    ----------
    .. [1] Choi, Park and Vogeley, (2007), astro-ph/0611607, doi:10.1086/511060

    """

    if size is None:
        size = np.broadcast(vd_min, vd_max, scale).shape or None

    lnx = np.linspace(vd_min, vd_max, resolution)

    def vdf_func(x):
        return 8e-3*(x/161)**2.32*np.exp(-(x/161)**2.67)*(2.67/special.gamma(2.32/2.67))*(1/x)

    pdf = vdf_func(lnx)
    cdf = pdf  # in place
    np.cumsum((pdf[1:]+pdf[:-1])/2*np.diff(lnx), out=cdf[1:])
    cdf[0] = 0
    cdf /= cdf[-1]

    t_lower = np.interp(vd_min, lnx, cdf)
    t_upper = np.interp(vd_max, lnx, cdf)

    u = np.random.uniform(t_lower, t_upper, size=size)
    lnx_sample = np.interp(u, cdf, lnx)

    return lnx_sample * scale
