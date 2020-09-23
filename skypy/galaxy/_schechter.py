'''Implementation of Schechter LF and SMF.'''

import numpy as np

from ..utils import uses_default_cosmology
from .redshift import schechter_lf_redshift
from .luminosity import schechter_lf_magnitude


__all__ = [
    'schechter_lf',
]


@uses_default_cosmology
def schechter_lf(redshift, M_star, phi_star, alpha, m_lim, fsky, cosmology, noise=True):
    r'''Sample redshifts and magnitudes from a Schechter luminosity function.

    Sample the redshifts and magnitudes of galaxies following a Schechter
    luminosity function with potentially redshift-dependent parameters, limited
    by an apparent magnitude `m_lim`, for a fraction `fsky` of the sky.

    Parameters
    ----------
    redshift : array_like
        Input redshift grid on which the Schechter function parameters are
        evaluated. Galaxies are sampled over this redshift range.
    M_star : array_like or function
        Characteristic absolute magnitude of the Schechter function. Can be a
        single value, an array of values for each `redshift`, or a function of
        redshift.
    phi_star : array_like or function
        Normalisation of the Schechter function. Can be a single value, an
        array of values for each `redshift`, or a function of redshift.
    alpha : array_like or function
        Schechter function power law index. Can be a single value, an array of
        values for each `redshift`, or a function of redshift.
    m_lim : float
        Limiting apparent magnitude.
    fsky : array_like
        Sky fraction over which galaxies are sampled.
    cosmology : Cosmology, optional
        Cosmology object to convert apparent to absolute magnitudes. If not
        given, the default cosmology is used.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Notes
    -----

    Effectively calls `~skypy.galaxy.redshift.schechter_lf_redshift` and
    `~skypy.galaxy.luminosity.schechter_lf_magnitude` internally and returns
    the tuple of results.

    Returns
    -------
    redshifts, magnitudes : tuple of array_like
        Redshifts and magnitudes of the galaxy sample described by the Schechter
        luminosity function.

    '''

    # sample galaxy redshifts
    z = schechter_lf_redshift(redshift, M_star, phi_star, alpha, m_lim, fsky, cosmology, noise)

    # if a function is NOT given for M_star, phi_star, alpha, interpolate to z
    if not callable(M_star) and np.ndim(M_star) > 0:
        M_star = np.interp(z, redshift, M_star)
    if not callable(phi_star) and np.ndim(phi_star) > 0:
        phi_star = np.interp(z, redshift, phi_star)
    if not callable(alpha) and np.ndim(alpha) > 0:
        alpha = np.interp(z, redshift, alpha)

    # sample galaxy magnitudes for redshifts
    M = schechter_lf_magnitude(z, M_star, alpha, m_lim, cosmology)

    return z, M
