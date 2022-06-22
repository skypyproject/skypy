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
   extincted_angle_schechter

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
                                  resolution=100):
    r'''axis ratio of a randomly projected triaxial ellipsoid obscured by a dust
    ring of maximum extinction E0.

    Given the two axis ratios `1 >= zeta >= xi` of a randomly oriented triaxial
    ellipsoid, computes the axis ratio `q` of the projection.
    The polar viewing angle is drawn from a non-flat cosine distribution
    following equation (3) in [2], assuming a Schechter luminosity function
    characterised by M_star and alpha and limited by M_lim.

    Parameters
    ----------
    zeta : array_like
        Axis ratio of intermediate and major axis.
    xi : array_like
        Axis ratio of minor and major axis.
    M_star : float
        Characteristic absolute magnitude of the Schechter function in [2]_.
    alpha : float or int
        The alpha parameter in the Schechter function in [2]_.
    E0 : float
        Edge-on extinction in magnitudes
    M_lim : float
        Absolute magnitude limit of the Schechter function in [2]_.
    resolution : int
        Resolution of the inverse transform sampling spline. Default is 1000.

    Returns
    -------
    q : (size,) array_like
        Axis ratio of the randomly projected ellipsoid.

    Notes
    -----
    See equations (11) and (12) in [1]_ and (1), (2) and (3) in [3]_
    for details.

    References
    ----------
    .. [1] Binney J., 1985, MNRAS, 212, 767. doi:10.1093/mnras/212.4.767
    .. [2] https://en.wikipedia.org/wiki/Luminosity_function_(astronomy)
    .. [3] Padilla N. D., Strauss M. A., 2008, MNRAS, 388, 1321

    '''

    size = np.broadcast(zeta, xi).shape

    # draw random viewing angle (theta, phi)
    # Ellipsoid with A > B > C axes. zeta = B/A; xi = C/A; y = C/B = xi/zeta
    y = np.divide(xi, zeta)
    theta = extincted_angle_schechter(y, E0, alpha, M_star, M_lim, resolution)
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


def extincted_angle_schechter(y, E0, alpha, M_star, M_lim, resolution=100):
    r'''Sample from dust extincted polar viewing angle distribution, assuming
    a Schechter luminosity function.

    Parameters
    ----------
    y : array_like
        Galaxy height to diameter ratio
    E0 : float
        Edge-on extinction in magnitudes
    alpha : float
        The alpha parameter in the Schechter function in [1]_.
    M_star : float
        Characteristic absolute magnitude of the Schechter function in [1]_.
    M_lim : float
        Absolute magnitude limit of the Schechter function in [1]_.
    resolution : int
        Resolution of the inverse transform sampling spline. Default is 1000.

    Returns
    -------
    theta_sample : (size,) array_like
        Polar viewing angle samples drawn from a non-flat extincted
        distribution.

    Warnings
    --------
    The inverse cumulative distribution function is approximated from the
    observed galaxy ratio psi evaluated on a spaced grid. The user
    must choose the `resolution` of this grid to satisfy their desired
    numerical accuracy.

    Notes
    -----
    See equations (1), (2) and (3) in [2]_ for details.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Luminosity_function_(astronomy)
    .. [2] Padilla N. D., Strauss M. A., 2008, MNRAS, 388, 1321

    '''

    # convert to array to get shape
    if np.isscalar(y):
        y = np.array([y])

    # get size from input ratios
    size = np.shape(y)
    length = y.size

    # generate array of y and theta values
    y = y.flatten()
    y = np.repeat(y, resolution, axis=-1)
    theta = np.tile(np.linspace(0, np.pi, resolution), length)

    # Extinction increases with theta. Eq (1) in [2]
    E_theta = np.ones_like(theta)*E0
    mask = np.abs(np.cos(theta)) > y
    E_theta[mask] = E0*(1 + y[mask] - np.abs(np.cos(theta[mask])))
    M_lim_ext = M_lim + E_theta

    # Convert to luminosities for easier integration
    L_lim = np.power(10, -0.4*M_lim)
    L_star = np.power(10, -0.4*M_star)
    L_lim_ext = np.power(10, -0.4*M_lim_ext)

    # Integration of the schechter function is an incomplete gamma function
    # Eq (3) in [2]
    psi = gammainc(alpha+1, L_lim_ext/L_star) / gammainc(alpha+1, L_lim/L_star)

    # Reshape arrays
    E_theta = E_theta.reshape(length, resolution)
    theta = theta.reshape(length, resolution)
    psi = psi.reshape(length, resolution)

    # Theta distribution is sine function, times the likelihood Psi
    pdf = np.sin(theta) * psi
    cdf = pdf  # in place
    np.cumsum((pdf[..., 1:]+pdf[..., :-1])/2*np.diff(theta), axis=1,
              out=cdf[..., 1:])
    cdf[:, 0] = 0
    norm = cdf[:, -1]
    cdf /= norm[:, None]

    # Sampling from inverse cumulative distribution
    invu = np.random.uniform(0, 1, size=length)
    theta_samples = np.empty(length)
    for iu, u in enumerate(invu):
        theta_samples[iu] = np.interp(u, cdf[iu], theta[iu])

    theta_samples = theta_samples.reshape(size)
    # convert to scalar when size of the array is 1
    if length == 1:
        theta_samples = np.float(theta_samples)
    return theta_samples
