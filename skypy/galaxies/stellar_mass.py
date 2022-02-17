"""Models of galaxy stellar mass.
"""

import numpy as np

from ..utils.random import schechter


__all__ = [
    'schechter_smf_mass',
    'schechter_smf_parameters',
]


def schechter_smf_mass(redshift, alpha, m_star, m_min, m_max, size=None,
                       resolution=1000):
    r""" Stellar masses following the Schechter mass function [1]_.

    Parameters
    ----------
    redshift : array_like
        Galaxy redshifts for which to sample magnitudes.
    alpha : float
        The alpha parameter in the Schechter stellar mass function.
    m_star : (nm,) array-like
        Characteristic stellar mass m_*.
    size: int, optional
         Output shape of stellar mass samples. If size is None and m_star
         is a scalar, a single sample is returned. If size is None and
         m_star is an array, an array of samples is returned with the same
         shape as m_star.
    m_min, m_max : float
        Lower and upper bounds for the stellar mass.
    resolution : int, optional
        Resolution of the inverse transform sampling spline. Default is 100.

    Returns
    -------
    stellar mass : (nm,) array_like
        Drawn stellar masses from the Schechter stellar mass function in units
        of the solar mass.

    Notes
    -----
    The stellar mass probability distribution (pdf) follows a Schechter
    profile of the form

    .. math::

        \Phi(M) = \frac{1}{M_*} \left(\frac{M}{M_*}\right)^\alpha
            \exp\left(-\frac{M}{M_*}\right) \;.

    From this pdf one can sample the stellar masses.

    References
    ----------
    .. [1] Mo, H., Van den Bosch, F., & White, S. (2010). Galaxy Formation and
        Evolution. Cambridge: Cambridge University Press.
        doi:10.1017/CBO9780511807244

    """

    # only alpha scalars supported at the moment
    if np.ndim(alpha) > 0:
        raise NotImplementedError('only scalar alpha is supported')

    if size is None and np.shape(redshift):
        size = np.shape(redshift)

    # convert m_min, m_max to units of m_star
    x_min = m_min / m_star
    x_max = m_max / m_star

    # sample masses
    m = schechter(alpha, x_min, x_max, resolution, size=size, scale=m_star)

    return m


def schechter_smf_parameters(active_parameters, fsatellite, fenvironment):
    r'''Schechter parameters.
    This function returns the Schechter mass function parameters
    for active galaxies (centrals and satellites)
    and passive galaxies (mass- and satellite-quenched)
    based on equation (15) in de la Bella et al. 2021 [1]_.

    Parameters
    ----------
    active_parameters: tuple
        Schechter mass function parameters for the entire active
        sample of galaxies: :math:`(\phi_b, \alpha_b, m_{*})`.

    fsatellite: float, (n, ) array_like
        Fraction of active satellite galaxies between 0 and 1.
        It could be a float or an array, depending on the model you choose.

    fenvironment: float
        Fraction of satellite-quenched galaxies between 0 and 1.


    Returns
    -------
    parameters: dic
        It returns a  dictionary with the parameters of the
        Schechter mass function. The dictionary keywords:
        `centrals`, `satellites`, `mass_quenched` and
        `satellite_quenched`. The values correspond to a
        tuple :math:`(\phi, \alpha, m_{*})`.

    References
    ----------
    .. [1] de la Bella et al. 2021, Quenching and Galaxy Demographics,
           arXiv 2112.11110.

    '''
    phi, alpha, mstar = active_parameters

    sum_phics = (1 - fsatellite) * (1 - np.log(1 - fsatellite))
    phic = (1 - fsatellite) * phi / sum_phics
    phis = phic * np.log(1 / (1 - fsatellite))

    centrals = (phic, alpha, mstar)
    satellites = (phis, alpha, mstar)
    mass_quenched = (phi, alpha + 1, mstar)
    satellite_quenched = (- np.log(1 - fenvironment) * phis, alpha, mstar)

    return {'centrals': centrals, 'satellites': satellites,
            'mass_quenched': mass_quenched, 'satellite_quenched': satellite_quenched}
