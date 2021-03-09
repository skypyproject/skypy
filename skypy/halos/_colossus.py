"""Colossus dark matter halo properties.

This module contains functions that interfaces with the external code
`Colossus <http://www.benediktdiemer.com/code/colossus/>`_.

"""

import numpy as np
from scipy import integrate

__all__ = [
    'colossus_mass_sampler',
]

try:
    import colossus
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
    -----------
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
    --------
    sample : (size,) array_like
        Samples drawn from the mass function, in units of solar masses.

        References
    -----------
    .. [1] Diemer B., 2018, ApJS, 239, 35

    """
    from colossus.lss import mass_function
    cosmo = colossus.cosmology.cosmology.fromAstropy(cosmology, sigma8=sigma8,
                                                     ns=ns, name='my_cosmo')
    h0 = cosmo.h
    m_h0 = np.logspace(np.log10(m_min*h0), np.log10(m_max*h0), resolution)  # unit: Msun/h
    dndm = mass_function.massFunction(m_h0, redshift, mdef=mdef, model=model,
                                      q_out='dndlnM', q_in='M')/m_h0
    m = m_h0/h0
    CDF = integrate.cumtrapz(dndm, (m), initial=0)
    CDF = CDF / CDF[-1]
    n_uniform = np.random.uniform(size=size)
    masssample = np.interp(n_uniform, CDF, m)
    return masssample
