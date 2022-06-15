"""Models of galaxy stellar mass.
"""

import numpy as np

from ..utils.random import schechter


__all__ = [
    'schechter_smf_mass',
    'schechter_smf_amplitude_centrals',
    'schechter_smf_amplitude_satellites',
    'schechter_smf_amplitude_mass_quenched',
    'schechter_smf_amplitude_satellite_quenched',
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


def schechter_smf_amplitude_centrals(phi_blue_total, fsatellite):
    r'''Schechter amplitude of centrals.
    This function returns the Schechter mass function amplitude
    for active central population based on equation (15)
    in de la Bella et al. 2021 [1]_.

    Parameters
    ----------
    phi_blue_total: float, (nt, ) array_like
        Schechter mass function amplitude for the entire active
        sample of galaxies, :math:`(\phi_b, \alpha_b, m_{*})`.

    fsatellite: float, (nm, ) array_like
        Fraction of active satellite galaxies between 0 and 1.
        It could be a float or an array, depending on the model you choose.

    Returns
    -------
    amplitude: float, (nt, nm) array_like
        Amplitude of the Schechter mass function.

     References
    ----------
    .. [1] de la Bella et al. 2021, Quenching and Galaxy Demographics,
           arXiv 2112.11110.

    '''

    if np.ndim(phi_blue_total) == 1 and np.ndim(fsatellite) == 1:
        phi_blue_total = phi_blue_total[:, np.newaxis]

    sum_phics = (1 - fsatellite) * (1 - np.log(1 - fsatellite))

    return (1 - fsatellite) * phi_blue_total / sum_phics


def schechter_smf_amplitude_satellites(phi_centrals, fsatellite):
    r'''Schechter amplitude of satellites.
    This function returns the Schechter mass function amplitude
    for active satellite population based on equation (15)
    in de la Bella et al. 2021 [1]_.

    Parameters
    ----------
    phi_centrals: float, (nt, nm) array_like
        Schechter mass function amplitude of the central
        active galaxies.

    fsatellite: float, (nm, ) array_like
        Fraction of active satellite galaxies between 0 and 1.
        It could be a float or an array, depending on the model you choose.

    Returns
    -------
    amplitude: float, (nt, nm) array_like
        Amplitude of the Schechter mass function.

    References
    ----------
    .. [1] de la Bella et al. 2021, Quenching and Galaxy Demographics,
           arXiv 2112.11110.

    '''

    if np.ndim(phi_centrals) == 2 and np.ndim(fsatellite) == 1:
        phi_centrals = phi_centrals[:, np.newaxis]

    return phi_centrals * np.log(1 / (1 - fsatellite))


def schechter_smf_amplitude_mass_quenched(phi_centrals, phi_satellites):
    r'''Schechter amplitude of satellites.
    This function returns the Schechter mass function amplitude
    for passive mass-quenched population based on equation (15)
    in de la Bella et al. 2021 [1]_.

    Parameters
    ----------
    phi_centrals: float, (nt, nm) array_like
        Schechter mass function amplitude of the central
        active galaxies.

    phi_satellites: float, (nt, nm) array_like
        Schechter mass function amplitude of the satellite
        active galaxies.

    Returns
    -------
    amplitude: float, (nt, nm) array_like
        Amplitude of the Schechter mass function.

    References
    ----------
    .. [1] de la Bella et al. 2021, Quenching and Galaxy Demographics,
           arXiv 2112.11110.

    '''

    return phi_centrals + phi_satellites


def schechter_smf_amplitude_satellite_quenched(phi_satellites, fenvironment):
    r'''Schechter amplitude of satellites.
    This function returns the Schechter mass function amplitude
    for active central population based on equation (15)
    in de la Bella et al. 2021 [1]_.

    Parameters
    ----------
    phi_satellites: float, (nt, nm) array_like
        Schechter mass function amplitude of the satellite
        active galaxies.

    fenvironment: float
        Fraction of satellite-quenched galaxies between 0 and 1.

    Returns
    -------
    amplitude: float, (nt, nm) array_like
        Amplitude of the Schechter mass function.

    References
    ----------
    .. [1] de la Bella et al. 2021, Quenching and Galaxy Demographics,
           arXiv 2112.11110.

    '''

    return - np.log(1 - fenvironment) * phi_satellites
