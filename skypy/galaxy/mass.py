r"""Models of galaxy luminosities.

"""

import numpy as np

from skypy.utils.random import schechter


__all__ = [
    'peng_masses',
]


def peng_masses(redshift, alpha, m_star, size=None,
                x_min=0.0002138, x_max=213.8, resolution=100):
    r"""Model of Peng et al (2010)

    Stellar masses following the Schechter mass function following the
    Peng et al. [1]_ model.

    Parameters
    ----------
    redshift : (nz,) array-like
        The redshift values at which to sample luminosities.
    alpha : float or int
        The alpha parameter in the Schechter stellar mass function.
    m_star : float or int
        Characteristic stellar mass M_*.
    size: int, optional
         Output shape of stellar mass samples. If size is None and redshift
         is a scalar, a single sample is returned. If size is None and
         redshift is an array, an array of samples is returned with the same
         shape as redshift.
    x_min, x_max : float or int, optional
        Lower and upper stella mass bounds in units of M_*.
    resolution : int, optional
        Resolution of the inverse transform sampling spline. Default is 100.

    Returns
    -------
    stellar mass : array_like
        Drawn stellar masses from the Schechter stellar mass function in units
        of the solar mass

    Notes
    -----
    The Schechter luminosity function is given as

    .. math::
        \Phi(M, z) = \frac{\Phi_\star(z)}{M_\star}
            \left(\frac{M}{M_\star}\right)^\alpha
            \exp\left(-\frac{M}{M_\star}\right) \;.

   The redshift dependence of :math:`\Phi_\star(z)` is given by

    .. math::

        \Phi_\star(z) = a_\phi z + b_phi \;.

    Now we have to rescale the Schechter function by the comoving element and
    get

    .. math::

        \phi(M,z) = \frac{d_H d_M^2}{E(z)}  \Phi(M,z)\;.

    References
    ----------
    .. [1] Peng Y., Lilly S. J. et al., 2010, The Astrophysical Journal,
    Issue 1, Volume 721, pages 193-221

    Examples
    --------
    >>> import skypy.galaxy.mass as mass

    Sample 100 stellar masses values at redshift z = 1.0 with alpha = -1.4, and
    m_star = 10**10.67.

    >>> masses = mass.peng_masses(1.0, -1.4, 10**10.67, size=100)

    Sample a luminosity value for every redshift in an array z with
    alpha = -1.3 amd m_star = 10**10.67.

    >>> z = np.linspace(0,2, 100)
    >>> masses = mass.peng_masses(z, -1.4, m_star = 10**10.67)



    """

    if size is None and np.shape(redshift):
        size = np.shape(redshift)

    x_sample = schechter(alpha, x_min, x_max, resolution=resolution, size=size)

    return m_star * x_sample
