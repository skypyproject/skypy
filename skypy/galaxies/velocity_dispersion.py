r"""Models of galaxy velocity dispersion.

"""

import numpy as np

__all__ = [
    'schechter_vdf',
]

def schechter_vdf(alpha, beta, vd_star, vd_min, vd_max, size=None, resolution=1000):
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

        {dn}={\phi }_{*}{(\frac{\sigma}{{\sigma}_{*}})}^{\alpha}
        {exp}[-{(\frac{\sigma}{{\sigma}_{*}})}^{\beta}]
        \frac{\beta}{{\Gamma}(\alpha/\beta)}\frac{1}{\sigma},

        where :math:`\sigma` is velocity dispersion, :math:`\sigma_*` is the charactersitic velocity dispersion, :math:`\phi_*` 
        is number density of all spiral galaxies and :math:`\alpha` and :math:`\beta` are free parameters.

    References
    ----------
    .. [1] Choi, Park and Vogeley, (2007), astro-ph/0611607, doi:10.1086/511060

    """

    # if np.ndim(alpha) > 0:
    #     raise NotImplementedError('only scalar alpha is supported')

    # alpha_prime = alpha - 1

    # samples = schechter(alpha, vd_min, vd_max, resolution = resolution, size = size)

    # samples = samples*sigma_star/beta

    # return samples
