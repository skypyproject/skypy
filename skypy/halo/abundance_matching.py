"""Abundance matching module.

This module provides methods to perform abundance matching between catalogs of
galaxies and dark matter halos.

Models
======

.. autosummary::
   :nosignatures:
   :toctree: ../api/

   vale_ostriker

"""

import numpy as np
from skypy.halo.mass import press_schechter, number_subhalos, subhalo_mass_sampler
from skypy.galaxy.luminosity import schechter_lf_magnitude

__all__ = [
    'vale_ostriker',
]


def vale_ostriker(halo_kwargs, subhalo_kwargs, galaxy_kwargs):
    """Vale & Ostriker abundance matching.
    Generate matched arrays of (sub)halos masses and galaxy absolute magnitudes
    following the abundance matching model in Vale & Ostriker (2004).

    Parameters
    ----------
    halo_kwargs : dict
        Dictionary of keyword arguments for skypy.halo.press_schechter.
    subhalo_kwargs : dict
        Dictionary of keyword arguments for skypy.halo.number_subhalos and
        skypy.halo.subhalo_mass_sampler.
    galaxy_kwargs : dict
        Dictionary of keyword arguments for skypy.galaxy.schechter_lf_magnitude.

    Returns
    -------
    mass : array_like
        Array of (sub)halo masses.
    magnitude : array_like
        Array of galaxy absolute magnitudes.
    group : array_like
        Array of halo group ids.
    parent : array_like
        Array of boolean values indicating if the halo is a parent.
    References
    ----------
    .. [1] Vale A., Ostriker J. P., 2004, MNRAS, 353, 189
    """

    # Sample halo and subhalo masses
    halo_mass = press_schechter(**halo_kwargs)
    halo_mass[::-1].sort()
    n_subhalos = number_subhalos(halo_mass, **subhalo_kwargs)
    del(subhalo_kwargs['gamma_M'])
    del(subhalo_kwargs['noise'])
    subhalo_mass = subhalo_mass_sampler(halo_mass, n_subhalos, **subhalo_kwargs)

    # Assign subhalos to groups with their parent halos
    n_halos = halo_kwargs.get('size')
    halo_group = np.arange(n_halos)
    indexing = np.zeros(n_halos + 1, dtype=int)
    indexing[1:] = np.cumsum(n_subhalos)
    total_subhalos = indexing[-1]
    subhalo_group = np.empty(total_subhalos)
    for first, last, id in zip(indexing[:-1], indexing[1:], halo_group):
        subhalo_group[first:last] = id

    # Concatenate halos and subhalos
    mass = np.concatenate([halo_mass, subhalo_mass])
    group = np.concatenate([halo_group, subhalo_group])
    parent = np.array((True,) * n_halos + (False,) * total_subhalos)

    # Sample galaxy magnitudes
    n_galaxies = galaxy_kwargs.get('size')
    magnitude = schechter_lf_magnitude(**galaxy_kwargs)

    # Sort halos and galaxies by mass and magnitude
    n_matches = min(n_halos + total_subhalos, n_galaxies)
    argsort_halos = np.argsort(mass)[-n_matches:][::-1]
    argsort_galaxies = np.argsort(magnitude)[:n_matches]

    return mass[argsort_halos], group[argsort_halos], parent[argsort_halos], magnitude[argsort_galaxies]
