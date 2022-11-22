r"""Models of galaxy velocity dispersion.

"""

import numpy as np
from skypy.utils.random import schechter

__all__ = [
    'schechter_vdf',
]


def schechter_vdf(alpha, beta, vd_star, vd_min, vd_max, size=None, resolution=1000):
    r"""Sample velocity dispersion of elliptical galaxies in the local universe
    following a Schecter function.

    Parameters
    ----------
    alpha: int
        The alpha parameter in the modified Schechter equation.
    beta: int
        The beta parameter in the modified Schechter equation.
    vd_star: int
        The characteristic velocity dispersion.
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

        \phi = \phi_* \left(\frac{\sigma}{\sigma_*}\right)^\alpha
            \exp\left[-\left( \frac{\sigma}{\sigma_*} \right)^\beta\right]
            \frac{\beta}{\Gamma(\alpha/\beta)} \frac{1}{\sigma} \mathrm{d}\sigma \;.

    where :math:`\Gamma` is the gamma function, :math:`\sigma_*` is the
    characteristic velocity dispersion, :math:`\phi_*` is
    number density of all spiral galaxies and
    :math:`\alpha` and :math:`\beta` are free parameters.

    References
    ----------
    .. [1] Choi, Park and Vogeley, (2007), astro-ph/0611607, doi:10.1086/511060

    """

    if np.ndim(alpha) > 0:
        raise NotImplementedError('only scalar alpha is supported')

    alpha_prime = alpha/beta - 1
    x_min, x_max = (vd_min/vd_star)**beta, (vd_max/vd_star)**beta

    samples = schechter(alpha_prime, x_min, x_max, resolution=resolution, size=size)
    samples = samples**(1/beta) * vd_star

    return samples
