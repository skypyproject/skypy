r"""Models of galaxy luminosities.


Models
======

.. autosummary::
   :nosignatures:
   :toctree: ../api/

   herbel_luminosities

"""

import numpy as np

import skypy.utils.astronomy as astro
from skypy.utils.random import schechter


def herbel_luminosities(redshift, alpha, a_m, b_m, size=None,
                        x_min=0.00305,
                        x_max=1100.0, resolution=100):

    r"""Model of Herbel et al (2017)

    Luminosities following the Schechter luminosity function following the
    Herbel et al. [1]_ model.

    Parameters
    ----------
    redshift : (nz,) array-like
        The redshift values at which to sample luminosities.
    alpha : float or int
        The alpha parameter in the Schechter luminosity function.
    a_m, b_m : float or int
        Factors parameterising the characteristic absolute magnitude M_* as
        a linear function of redshift according to Equation 3.3 in [1]_.
    size: int, optional
         Output shape of luminosity samples. If size is None and redshift
         is a scalar, a single sample is returned. If size is None and
         redshift is an array, an array of samples is returned with the same
         shape as redshift.
    x_min, x_max : float or int, optional
        Lower and upper luminosity bounds in units of L*.
    resolution : int, optional
        Resolution of the inverse transform sampling spline. Default is 100.

    Returns
    -------
    luminosity : array_like
        Drawn luminosities from the Schechter luminosity function.

    Notes
    -----
    The Schechter luminosity function is given as

    .. math::
        \Phi(L, z) = \frac{\Phi_\star(z)}{L_\star(z)}
            \left(\frac{L}{L_\star(z)}\right)^\alpha
            \exp\left(-\frac{L}{L_\star(z)}\right) \;.

    Here the luminosity is defined as

    .. math::

        L = 10^{-0.4M} \;,

    with absolute magnitude :math:`M`. Furthermore, Herbel et al. [1]_
    introduced

    .. math::

        \Phi_\star(z) = b_\phi \exp(a_\phi z) \;,
        M_\star(z) = a_M z + b_M \;.

    Now we have to rescale the Schechter function by the comoving element and
    get

    .. math::

        \phi(L,z) = \frac{d_H d_M^2}{E(z)}  \Phi(L,z)\;.

    References
    ----------
    .. [1] Herbel J., Kacprzak T., Amara A. et al., 2017, Journal of Cosmology
        and Astroparticle Physics, Issue 08, article id. 035 (2017)

    Examples
    --------
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



    """

    if size is None and np.shape(redshift):
        size = np.shape(redshift)

    luminosity_star = _calculate_luminosity_star(redshift, a_m, b_m)

    x_sample = schechter(alpha, x_min, x_max, resolution=resolution, size=size)

    return luminosity_star * x_sample


def _calculate_luminosity_star(redshift, a_m, b_m):
    absolute_magnitude_star = a_m * redshift + b_m
    return astro.luminosity_from_absolute_magnitude(absolute_magnitude_star)
