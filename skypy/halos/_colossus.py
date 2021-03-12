"""Colossus dark matter halo properties.

This module contains functions that interfaces with the external code
`Colossus <http://www.benediktdiemer.com/code/colossus/>`_.

"""

from astropy.cosmology import z_at_value
from astropy import units
import numpy as np
from scipy import integrate
from skypy.galaxies.redshift import redshifts_from_comoving_density

__all__ = [
    'colossus_mass_sampler',
]

try:
    import colossus  # noqa F401
except ImportError:
    HAS_COLOSSUS = False
else:
    HAS_COLOSSUS = True


def colossus_mass_sampler(redshift, model, mdef, m_min, m_max,
                          cosmology, sigma8, ns, size=None, resolution=1000):
    """Colossus halo mass sampler.

    This function generate a sample of halos from a mass function which
    is available in COLOSSUS [1]_.

    Parameters
    ----------
    redshift : float
        The redshift values at which to sample the halo mass.
    model : string
        Mass function model which is available in colossus.
    mdef : str
        Halo mass definition for spherical overdensities used by colossus.
    m_min, m_max : float
        Lower and upper bounds for halo mass in units of Solar mass, Msun.
    cosmology : astropy.cosmology.Cosmology
        Astropy cosmology object
    sigma8 : float
        Cosmology parameter, amplitude of the (linear) power spectrum on the
        scale of 8 Mpc/h.
    ns : float
        Cosmology parameter, spectral index of scalar perturbation power spectrum.
    size : int, optional
        Number of halos to sample. If size is None (default), a single value is returned.
    resolution : int, optional
        Resolution of the inverse transform sampling spline. Default is 1000.

    Returns
    -------
    sample : (size,) array_like
        Samples drawn from the mass function, in units of solar masses.

    References
    ----------
    .. [1] Diemer B., 2018, ApJS, 239, 35

    """
    from colossus.cosmology.cosmology import fromAstropy
    from colossus.lss import mass_function
    fromAstropy(cosmology, sigma8=sigma8, ns=ns)
    h0 = cosmology.h
    m_h0 = np.logspace(np.log10(m_min*h0), np.log10(m_max*h0), resolution)  # unit: Msun/h
    dndm = mass_function.massFunction(m_h0, redshift, mdef=mdef, model=model,
                                      q_out='dndlnM', q_in='M')/m_h0
    m = m_h0/h0
    CDF = integrate.cumtrapz(dndm, (m), initial=0)
    CDF = CDF / CDF[-1]
    n_uniform = np.random.uniform(size=size)
    masssample = np.interp(n_uniform, CDF, m)
    return masssample


@units.quantity_input(sky_area=units.sr)
def colossus_mf_redshift(redshift, model, mdef, m_min, m_max, sky_area,
                         cosmology, sigma8, ns, resolution=1000, noise=True):
    r'''Sample redshifts from a COLOSSUS halo mass function.

    Sample the redshifts of dark matter halos following a mass function
    implemented in COLOSSUS [1]_ within given mass and redshift ranges and for
    a given area of the sky.

    Parameters
    ----------
    redshift : array_like
        Input redshift grid on which the mass function is evaluated. Halos are
        sampled over this redshift range.
    model : string
        Mass function model which is available in colossus.
    mdef : str
        Halo mass definition for spherical overdensities used by colossus.
    m_min, m_max : float
        Lower and upper bounds for the halo mass in units of Solar mass, Msun.
    sky_area : `~astropy.units.Quantity`
        Sky area over which halos are sampled. Must be in units of solid angle.
    cosmology : Cosmology
        Cosmology object to convert comoving density.
    sigma8 : float
        Cosmology parameter, amplitude of the (linear) power spectrum on the
        scale of 8 Mpc/h.
    ns : float
        Cosmology parameter, spectral index of scalar perturbation power
        spectrum.
    noise : bool, optional
        Poisson-sample the number of halos. Default is `True`.

    Returns
    -------
    redshifts : array_like
        Redshifts of the halo sample.

    References
    ----------
    .. [1] Diemer B., 2018, ApJS, 239, 35

    '''
    from colossus.cosmology.cosmology import fromAstropy
    from colossus.lss.mass_function import massFunction

    # Set the cosmology in COLOSSUS
    fromAstropy(cosmology, sigma8, ns)

    # Integrate the mass function to get the number density of halos at each redshift
    def dndlnM(lnm, z):
        return massFunction(np.exp(lnm), z, 'M', 'dndlnM', mdef, model)
    lnmmin = np.log(m_min/cosmology.h)
    lnmmax = np.log(m_max/cosmology.h)
    density = [integrate.quad(dndlnM, lnmmin, lnmmax, args=(z))[0] for z in redshift]
    density = np.array(density) * np.power(cosmology.h, 3)

    # Sample halo redshifts and assign to bins
    return redshifts_from_comoving_density(redshift, density, sky_area, cosmology, noise)


@units.quantity_input(sky_area=units.sr)
def colossus_mf(redshift, model, mdef, m_min, m_max, sky_area, cosmology,
                sigma8, ns, resolution=1000, noise=True):
    r'''Sample halo redshifts and masses from a COLOSSUS mass function.

    Sample the redshifts and masses of dark matter halos following a mass
    function implemented in COLOSSUS [1]_ within given mass and redshift ranges
    and for a given area of the sky.

    Parameters
    ----------
    redshift : array_like
        Defines the edges of a set of redshift bins for which the mass function
        is evaluated. Must be a monotonically-increasing one-dimensional array
        of values. Halo redshifts are sampled between the minimum and maximum
        values in this array.
    model : string
        Mass function model which is available in colossus.
    mdef : str
        Halo mass definition for spherical overdensities used by colossus.
    m_min, m_max : float
        Lower and upper bounds for the halo mass in units of Solar mass, Msun.
    sky_area : `~astropy.units.Quantity`
        Sky area over which halos are sampled. Must be in units of solid angle.
    cosmology : Cosmology
        Cosmology object to calculate comoving densities.
    sigma8 : float
        Cosmology parameter, amplitude of the (linear) power spectrum on the
        scale of 8 Mpc/h.
    ns : float
        Cosmology parameter, spectral index of scalar perturbation power
        spectrum.
    noise : bool, optional
        Poisson-sample the number of halos. Default is `True`.

    Returns
    -------
    redshift, mass : tuple of array_like
        Redshifts and masses of the halo sample.

    References
    ----------
    .. [1] Diemer B., 2018, ApJS, 239, 35

    '''

    # Sample halo redshifts and assign to bins
    z = colossus_mf_redshift(redshift, model, mdef, m_min, m_max, sky_area,
                             cosmology, sigma8, ns, resolution, noise)
    redshift_bin = np.digitize(z, redshift)

    # Calculate the redshift at the centre of each bin
    comoving_distance = cosmology.comoving_distance(redshift)
    d_mid = 0.5 * (comoving_distance[:-1] + comoving_distance[1:])
    z_mid = [z_at_value(cosmology.comoving_distance, d) for d in d_mid]

    # Sample halo masses in each redshift bin
    m = np.empty_like(z)
    for i, zm in enumerate(z_mid):
        mask = redshift_bin == i + 1
        size = np.count_nonzero(mask)
        m[mask] = colossus_mass_sampler(zm, model, mdef, m_min, m_max,
                                        cosmology, sigma8, ns, size, resolution)

    return z, m
