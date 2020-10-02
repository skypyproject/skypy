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
        Array of (sub)halo masses in units of solar masses.
    group : array_like
        Array of halo group ID numbers.
    parent : array_like
        Array of boolean values indicating if the halo is a parent.
    magnitude : array_like
        Array of galaxy absolute magnitudes.
    References
    ----------
    .. [1] Vale A., Ostriker J. P., 2004, MNRAS, 353, 189
    """

    # Sample halo and subhalo masses
    halo_mass = press_schechter(**halo_kwargs)
    halo_mass[::-1].sort()  # Sort in-place from high to low for indexing
    n_subhalos = number_subhalos(halo_mass, **subhalo_kwargs)
    sampler_kwargs = {k: v for k, v in subhalo_kwargs.items() if k not in ['gamma_M', 'noise']}
    subhalo_mass = subhalo_mass_sampler(halo_mass, n_subhalos, **sampler_kwargs)

    # Assign subhalos to groups with their parent halos
    n_halos = len(halo_mass)
    halo_group = np.arange(n_halos)
    indexing = np.zeros(n_halos + 1, dtype=int)
    indexing[1:] = np.cumsum(n_subhalos)
    total_subhalos = indexing[-1]
    subhalo_group = np.empty(total_subhalos, dtype=int)
    for first, last, id in zip(indexing[:-1], indexing[1:], halo_group):
        subhalo_group[first:last] = id

    # Concatenate halos and subhalos
    mass = np.concatenate([halo_mass, subhalo_mass])
    group = np.concatenate([halo_group, subhalo_group])
    parent = np.empty(n_halos+total_subhalos, dtype=bool)
    parent[:n_halos] = True
    parent[n_halos:] = False

    # Sample galaxy magnitudes
    magnitude = schechter_lf_magnitude(**galaxy_kwargs)
    n_galaxies = len(magnitude)

    # Sort halos and galaxies by mass and magnitude
    n_matches = min(n_halos + total_subhalos, n_galaxies)
    sort_mass = np.argsort(mass)[-n_matches:][::-1]
    sort_magnitudes = np.argsort(magnitude)[:n_matches]

    mass = mass[sort_mass]
    group = group[sort_mass]
    parent = parent[sort_mass]
    magnitude = magnitude[sort_magnitudes]

    return mass, group, parent, magnitude
