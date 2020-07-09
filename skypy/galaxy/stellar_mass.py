r"""Models of galaxy mass (total and stellar).

"""

import numpy as np

from skypy.utils.random import schechter


__all__ = [
    'stellar_mass_function',
]


def stellar_mass_function(alpha, m_star, size=None,
                          x_min=0.0002138, x_max=213.8, resolution=100):
    r""" Stellar masses following the Schechter mass function.

    Parameters
    ----------
    alpha : float or int
        The alpha parameter in the Schechter stellar mass function.
    m_star : array-like
        Characteristic stellar mass M_*.
    size: int, optional
         Output shape of stellar mass samples. If size is None and m_star
         is a scalar, a single sample is returned. If size is None and
         m_star is an array, an array of samples is returned with the same
         shape as m_star.
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
     The stellar mass function is given by a Schechter function as

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

    Examples
    --------
    >>> import skypy.galaxy.stellar_mass as mass

    Sample 100 stellar masses values at redshift z = 1.0 with alpha = -1.4, and
    m_star = 10**10.67.

    >>> masses = mass.stellar_mass_function(1.0, -1.4, 10**10.67, size=100)

    Sample a luminosity value for every redshift in an array z with
    alpha = -1.3 amd m_star = 10**10.67.

    >>> z = np.linspace(0,2, 100)
    >>> masses = mass.stellar_mass_function(z, -1.4, m_star = 10**10.67)



    """

    if size is None and np.shape(m_star):
        size = np.shape(m_star)

    x_sample = schechter(alpha, x_min, x_max, resolution=resolution, size=size)

    return m_star * x_sample
