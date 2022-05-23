"""Random utility module.

This module provides methods to draw from random distributions.


Utility Functions
=================

.. autosummary::
   :nosignatures:
   :toctree: ../api/

   schechter
   triaxial_axis_ratio
   triaxial_axis_ratio_extincted

"""

import numpy as np
from scipy.special import gammainc


def schechter(alpha, x_min, x_max, resolution=100, size=None, scale=1.):
    """Sample from the Schechter function.

    Parameters
    ----------
    alpha : float or int
        The alpha parameter in the Schechter function in [1]_.
    x_min, x_max : array_like
        Lower and upper bounds for the random variable x.
    resolution : int
        Resolution of the inverse transform sampling spline. Default is 100.
    size: int, optional
        Output shape of samples. If size is None and scale is a scalar, a
        single sample is returned. If size is None and scale is an array, an
        array of samples is returned with the same shape as scale.
    scale: array-like, optional
        Scale factor for the returned samples. Default is 1.

    Returns
    -------
    x_sample : array_like
        Samples drawn from the Schechter function.

    Warnings
    --------
    The inverse cumulative distribution function is approximated from the
    Schechter function evaluated on a logarithmically-spaced grid. The user
    must choose the `resolution` of this grid to satisfy their desired
    numerical accuracy.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Luminosity_function_(astronomy)
    """

    if size is None:
        size = np.broadcast(x_min, x_max, scale).shape or None

    lnx_min = np.log(x_min)
    lnx_max = np.log(x_max)

    lnx = np.linspace(np.min(lnx_min), np.max(lnx_max), resolution)

    pdf = np.exp(np.add(alpha, 1)*lnx - np.exp(lnx))
    cdf = pdf  # in place
    np.cumsum((pdf[1:]+pdf[:-1])/2*np.diff(lnx), out=cdf[1:])
    cdf[0] = 0
    cdf /= cdf[-1]

    t_lower = np.interp(lnx_min, lnx, cdf)
    t_upper = np.interp(lnx_max, lnx, cdf)
    u = np.random.uniform(t_lower, t_upper, size=size)
    lnx_sample = np.interp(u, cdf, lnx)

    return np.exp(lnx_sample) * scale


def triaxial_axis_ratio(zeta, xi, size=None):
    r'''axis ratio of a randomly projected triaxial ellipsoid

    Given the two axis ratios `1 >= zeta >= xi` of a randomly oriented triaxial
    ellipsoid, computes the axis ratio `q` of the projection.

    Parameters
    ----------
    zeta : array_like
        Axis ratio of intermediate and major axis.
    xi : array_like
        Axis ratio of minor and major axis.
    size : tuple of int or None
        Size of the random draw. If `None` is given, size is inferred from
        other inputs.

    Returns
    -------
    q : array_like
        Axis ratio of the randomly projected ellipsoid.

    Notes
    -----
    See equations (11) and (12) in [1]_ for details.

    References
    ----------
    .. [1] Binney J., 1985, MNRAS, 212, 767. doi:10.1093/mnras/212.4.767
    '''

    # get size from inputs if not explicitly provided
    if size is None:
        size = np.broadcast(zeta, xi).shape

    # draw random viewing angle (theta, phi)
    cos2_theta = np.random.uniform(low=-1., high=1., size=size)
    cos2_theta *= cos2_theta
    sin2_theta = 1 - cos2_theta
    cos2_phi = np.cos(np.random.uniform(low=0., high=2*np.pi, size=size))
    cos2_phi *= cos2_phi
    sin2_phi = 1 - cos2_phi

    # transform arrays to quantities that are used in eq. (11)
    z2m1 = np.square(zeta)
    z2m1 -= 1
    x2 = np.square(xi)

    # eq. (11) multiplied by xi^2 zeta^2
    A = (1 + z2m1*sin2_phi)*cos2_theta + x2*sin2_theta
    B2 = 4*z2m1**2*cos2_theta*sin2_phi*cos2_phi
    C = 1 + z2m1*cos2_phi

    # eq. (12)
    q = np.sqrt((A+C-np.sqrt((A-C)**2+B2))/(A+C+np.sqrt((A-C)**2+B2)))

    return q

def triaxial_axis_ratio_extincted(zeta, xi, M_star, alpha, E0, M_lim,
                                  size=None):
    r'''axis ratio of a randomly projected triaxial ellipsoid

    Given the two axis ratios `1 >= zeta >= xi` of a randomly oriented triaxial
    ellipsoid, computes the axis ratio `q` of the projection.

    Parameters
    ----------
    zeta : array_like
        Axis ratio of intermediate and major axis.
    xi : array_like
        Axis ratio of minor and major axis.
    size : tuple of int or None
        Size of the random draw. If `None` is given, size is inferred from
        other inputs.

    Returns
    -------
    q : array_like
        Axis ratio of the randomly projected ellipsoid.

    Notes
    -----
    See equations (11) and (12) in [1]_ for details.

    References
    ----------
    .. [1] Binney J., 1985, MNRAS, 212, 767. doi:10.1093/mnras/212.4.767
    '''

    # get size from inputs if not explicitly provided
    if size is None:
        size = np.broadcast(zeta, xi).shape

    # draw random viewing angle (theta, phi)
    # Ellipsoid with A > B > C axes. zeta = B/A; xi = C/A; y = C/B = xi/zeta
    y = xi/zeta
    theta = schecter_extincted_angle(y, E0, alpha, M_star, M_lim, resolution,
                                     size=size)
    cos2_theta = np.cos(theta)**2

    sin2_theta = 1 - cos2_theta
    cos2_phi = np.cos(np.random.uniform(low=0., high=2*np.pi, size=size))
    cos2_phi *= cos2_phi
    sin2_phi = 1 - cos2_phi

    # transform arrays to quantities that are used in eq. (11)
    z2m1 = np.square(zeta)
    z2m1 -= 1
    x2 = np.square(xi)

    # eq. (11) multiplied by xi^2 zeta^2
    A = (1 + z2m1*sin2_phi)*cos2_theta + x2*sin2_theta
    B2 = 4*z2m1**2*cos2_theta*sin2_phi*cos2_phi
    C = 1 + z2m1*cos2_phi

    # eq. (12)
    q = np.sqrt((A+C-np.sqrt((A-C)**2+B2))/(A+C+np.sqrt((A-C)**2+B2)))

    return q

def schecter_extincted_angle(y, E0, alpha, M_star, M_lim, resolution,
                             size = None):
    """
    This function should return an array of theta angles use to project a
    triaxial ellipsoid. The distribution depends on the parameters of the
    schecter luminosity function
    """

    # Extinction increases with theta. Eq (1)
    theta = np.linspace(0, np.pi, resolution)
    theta_blocked = np.where(np.cos(theta) > y)
    E_theta = np.ones_like(theta)*E0
    E_theta[theta_blocked] = E0*(1 + y - np.cos(theta)[theta_blocked])

    # Extincted magnitude limit as a function of theta
    M_lim_ext = M_lim + E_theta

    # Convert to luminosities for easier integration
    L_lim = 10**(-0.4*M_lim)
    L_star = 10**(-0.4*M_star)
    L_lim_ext = 10**(-0.4*M_lim_ext)

    # Integration of the schecter function is an incomplete gamma function
    phi = gammainc(alpha+1, L_lim_ext/L_star) / gammainc(alpha+1, L_lim/L_star)
