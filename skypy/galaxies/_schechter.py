'''Implementation of Schechter LF and SMF.'''

import numpy as np

from .redshift import schechter_lf_redshift, schechter_smf_redshift
from .stellar_mass import schechter_smf_mass
from .luminosity import schechter_lf_magnitude
from astropy import units

__all__ = [
    'schechter_lf',
    'schechter_smf',
]


@units.quantity_input(sky_area=units.sr)
def schechter_lf(redshift, M_star, phi_star, alpha, m_lim, sky_area, cosmology, noise=True):
    r'''Sample redshifts and magnitudes from a Schechter luminosity function.

    Sample the redshifts and magnitudes of galaxies following a Schechter
    luminosity function with potentially redshift-dependent parameters, limited
    by an apparent magnitude `m_lim`, for a sky area `sky_area`.

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
    sky_area : `~astropy.units.Quantity`
        Sky area over which galaxies are sampled. Must be in units of solid angle.
    cosmology : Cosmology
        Cosmology object to convert apparent to absolute magnitudes.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Notes
    -----

    Effectively calls `~skypy.galaxies.redshift.schechter_lf_redshift` and
    `~skypy.galaxies.luminosity.schechter_lf_magnitude` internally and returns
    the tuple of results.

    Returns
    -------
    redshifts, magnitudes : tuple of array_like
        Redshifts and magnitudes of the galaxy sample described by the Schechter
        luminosity function.

    '''

    # sample galaxy redshifts
    z = schechter_lf_redshift(redshift, M_star, phi_star, alpha, m_lim, sky_area, cosmology, noise)

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


@units.quantity_input(sky_area=units.sr)
def schechter_smf(redshift, m_star, phi_star, alpha, m_min, m_max, sky_area, cosmology, noise=True):
    r'''Sample redshifts and stellar masses from a Schechter mass function.

    Sample the redshifts and stellar masses of galaxies following a Schechter
    mass function with potentially redshift-dependent parameters, limited
    by maximum and minimum masses `m_min`, `m_max`, for a sky area `sky_area`.

    Parameters
    ----------
    redshift : array_like
        Input redshift grid on which the Schechter function parameters are
        evaluated. Galaxies are sampled over this redshift range.
    m_star : array_like or function
        Characteristic mass of the Schechter function. Can be a
        single value, an array of values for each `redshift`, or a function of
        redshift.
    phi_star : array_like or function
        Normalisation of the Schechter function. Can be a single value, an
        array of values for each `redshift`, or a function of redshift.
    alpha : array_like or function
        Schechter function power law index. Can be a single value, an array of
        values for each `redshift`, or a function of redshift.
    m_min, m_max : float
        Lower and upper bounds for the stellar mass.
    sky_area : `~astropy.units.Quantity`
        Sky area over which galaxies are sampled. Must be in units of solid angle.
    cosmology : Cosmology
        Cosmology object to calculate comoving densities.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Notes
    -----

    Effectively calls `~skypy.galaxies.redshift.schechter_smf_redshift` and
    `~skypy.galaxies.stellar_mass.schechter_smf_mass` internally and returns
    the tuple of results.

    Returns
    -------
    redshifts, stellar masses : tuple of array_like
        Redshifts and stellar masses of the galaxy sample described by the Schechter
        stellar mass function.

    '''

    # sample halo redshifts
    z = schechter_smf_redshift(redshift, m_star, phi_star, alpha, m_min, m_max,
                               sky_area, cosmology, noise)

    # if a function is NOT given for M_star, phi_star, alpha, interpolate to z
    if not callable(m_star) and np.ndim(m_star) > 0:
        m_star = np.interp(z, redshift, m_star)
    if not callable(phi_star) and np.ndim(phi_star) > 0:
        phi_star = np.interp(z, redshift, phi_star)
    if not callable(alpha) and np.ndim(alpha) > 0:
        alpha = np.interp(z, redshift, alpha)

    # sample galaxy mass for redshifts
    m = schechter_smf_mass(z, alpha, m_star, m_min, m_max)

    return z, m
