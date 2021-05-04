import numpy as np
from astropy import units


r"""HI mass module.
This module provides functions to calculate neutral Hydrogen masses for haloes.

"""

__all__ = [
           'pr_halo_model',
]


def pr_halo_model(halo_mass, circular_velocity, cosmology,
                  alpha=0.17,
                  beta=-0.55,
                  v_c0=np.exp(1.57) * (units.km / units.s),
                  v_c1=np.exp(4.39) * (units.km / units.s),
                  Y_p=0.24):
    r"""Model of Padmanabhan & Refregier (2017), equation 1.

    Halo neutral Hydrogen (HI 21-cm) masses as a function of halo mass,
    halo circular velocity, and cosmology.

    Parameters
    ----------
    halo_mass : (nhalos,) `~astropy.Quantity`
        The masses of the underlying halos in units of solar mass.
    circular_velocity : (nhalos,) `~astropy.Quantity`
        The circular velocity for the halos in units of [km s-1].
    cosmology : `~astropy.cosmology.Cosmology`
        Cosmology object providing values of omega_baryon and omega_matter.
    alpha : float
        Linear model constant, fit from data.
    beta : float
        Model exponent, fit from data.
    v_c0 : `~astropy.Quantity`
        Velocity for the lower exponential cutoff of the model
        in units of [km s-1].
    v_c1 : `~astropy.Quantity`
        Velocity for the upper exponential cutoff of the model
        in units of [km s-1].
    Y_p : float
        Cosmic Helium fraction.

    Returns
    -------
    m_hone : (nhalos,) `~astropy.Quantity`
        Neutral hydrogen mass contained in the halo, in units of solar mass.

    References
    ----------
    .. Padmanabhan & Refregier 2017, MNRAS, Volume 464, Issue 4, p. 4008
        https://arxiv.org/abs/1607.01021

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units
    >>> from astropy.cosmology import Planck15
    >>> from skypy.neutral_hydrogen import mass

    Sample halo masses on a grid and use a simple model for the circular velocity.

    >>> m_halo = np.logspace(8,15,128) * units.Msun
    >>> v_halo = (96.6 * (units.km / units.s)) * m_halo / (1.e11 * units.Msun)

    Calculate the neutral Hydrogen mass within each halo.

    >>> m_hone = pr_halo_model(m_halo, v_halo, Planck15)

    """

    f_Hc = 1. - Y_p * (cosmology.Ob0 / cosmology.Om0)

    lower_cutoff = np.exp(-(v_c0 / circular_velocity)**3.)
    upper_cutoff = np.exp(-(circular_velocity / v_c1)**3.)

    m_pivot = (1.e11 / cosmology.h) * units.Msun

    m_hone = (alpha * f_Hc * halo_mass * np.power(halo_mass / m_pivot, beta) *
              lower_cutoff * upper_cutoff)

    return m_hone
