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

from astropy.table import hstack, vstack


def vale_ostriker(halos, galaxies, subhalos=None,
                  mass='mass', luminosity='luminosity', join_type='inner'):
    """Vale & Ostriker abundance matching.
    Takes catalogs of (sub)halos and galaxies and performs abundance matching
    following the method outlined in Vale & Ostriker (2004) assuming
    monotonicity between halo mass and galaxy luminosity to return an
    abundance-matched table of halo-galaxy pairs.

    Parameters
    ----------
    halos, galaxies : Astropy Table
        Tables of halos and galaxies to be matched.
    subhalos : Astropy Table
        Optional table of subhalos to be matched..
    mass, luminosity : str
        Halo mass and galaxy luminosity column names.
    join_type : str
        Join type (‘inner’ | ‘exact’ | ‘outer’), default is inner

    Returns
    -------
    matched_Table : Astropy Table
        Table of abundance-matched halos and galaxies.

    References
    ----------
    .. [1] Vale A., Ostriker J. P., 2004, MNRAS, 353, 189
    """

    if subhalos:
        if mass not in halos.columns:
            raise ValueError("{} is not a column in halos".format(mass))
        if mass not in subhalos.columns:
            raise ValueError("{} is not a column in subhalos".format(mass))
        halos = vstack((halos, subhalos))

    halos.sort(mass, reverse=True)
    galaxies.sort(luminosity, reverse=True)
    return hstack([halos, galaxies], join_type=join_type)
