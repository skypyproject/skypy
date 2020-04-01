import numpy as np

import skypy.utils.astronomy as astro
import skypy.utils.special as special


def herbel_luminosities(redshift, alpha, a_m, b_m, size=None,
                        q_min=0.00305,
                        q_max=1100.0, resolution=100):

    r""" Luminosities following the Schechter luminosity function followjng the
        Herbel er al. (2017) model.

    Parameters
    ----------
    redshift : (nz,) array-like
        The redshift values at which to sample luminosities.
    alpha : float or int
        The alpha parameter in the Schechter luminosity function.
    a_m, b_m : float or int
        Factors parameterising the characteristic absolute magnitude M_* as
        a linear function of redshift according to Equation 3.3 in [1].
    size: int, optional
         Output shape of luminosity samples. If size is None and redshift
         is a scalar, a single sample is returned. If size is None and
         redshift is an array, an array of samples is returned with the same
         shape as redshift.
    q_min, q_max : float or int, optional
        Lower and upper luminosity bounds in units of L*.
    resolution : int, optional
        Resolution of the inverse transform sampling spline. Default is 100.

    Returns
    -------
    luminosity : array_like
        Drawn luminosities from the Schechter luminosity function.

    Notes
    -------
     The Schechter luminosity function is given as

    .. math::

        \Phi(L, z) = \frac{\Phi_\star(z)}{L_\star(z)}
            \left(\frac{L}{L_\star(z)}\right)^\alpha
            /exp\left(-\frac{L}{L_\star(z)}\right) \;.

    Here the luminosity is defined as

    .. math::

        L = 10^{-0.4M} \;,

    with absolute magnitude :math:`M`. Furthermore, Herbel et al. (2017)
    introduced

    .. math::

        \Phi_\star(z) = b_\phi \exp(a_\phi z) \;,
        M_\star(z) = a_M z + b_M \;.

    Now we have to rescale the Schechter function by the comoving element and
    get

    .. math::

        \phi(L,z) = \frac{d_H d_M^2}{E(z)}  \Phi(L,z)\;.

    Examples
    -------
    >>> import skypy.galaxy.luminosity as lum

    Sample 100 luminosity values at redshift z = 1.0 with
    a_m = -0.9408582, b_m = -20.40492365, alpha = -1.3.

    >>> luminosities = lum.herbel_luminosities(1.0, -1.3, -0.9408582,
    ...                                         -20.40492365, size=100)

    Sample a luminosity value for every redshift in an array z with
    a_m = -0.9408582, b_m = -20.40492365, alpha = -1.3.

    >>> z = np.linspace(0,2, 100)
    >>> luminosities = lum.herbel_luminosities(z, -1.3, -0.9408582,
    ...                                         -20.40492365)

    References
    -------
    [1] Herbel J., Kacprzak T., Amara A. et al., 2017, Journal of Cosmology and
    Astroparticle Physics, Issue 08, article id. 035 (2017)

    """

    if size is None and np.shape(redshift):
        size = np.shape(redshift)

    luminosity_star = _calculate_luminosity_star(redshift, a_m, b_m)
    q = np.logspace(np.log10(np.min(q_min)), np.log10(np.max(q_max)),
                    resolution)
    cdf = _cdf(q, np.min(q_min), np.max(q_max), alpha)
    t_lower = np.interp(q_min, q, cdf)
    t_upper = np.interp(q_max, q, cdf)
    u = np.random.uniform(t_lower, t_upper, size=size)
    q_sample = np.interp(u, cdf, q)
    luminosity_sample = luminosity_star * q_sample
    return luminosity_sample


def _cdf(q, q_min, q_max, alpha):
    a = special.upper_incomplete_gamma(alpha+1, q_min)
    b = special.upper_incomplete_gamma(alpha+1, q)
    c = special.upper_incomplete_gamma(alpha+1, q_min)
    d = special.upper_incomplete_gamma(alpha+1, q_max)
    return (a-b)/(c-d)


def _calculate_luminosity_star(redshift, a_m, b_m):
    absolute_magnitude_star = a_m * redshift + b_m
    return astro.luminosity_from_absolute_magnitude(absolute_magnitude_star)
