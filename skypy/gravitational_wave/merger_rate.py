
r"""Binary merger rate module.
This module provides functions to calculate compact binary merger rates
for individual galaxies.

"""

from astropy import constants, units


__all__ = [
    'b_band_merger_rate',
]


abadie_table_III = {
    'NS-NS': {
        'low': 0.6,
        'realistic': 60,
        'high': 600,
        'max': 2000},
    'NS-BH': {
        'low': 0.03,
        'realistic': 2,
        'high': 60},
    'BH-BH': {
        'low': 0.006,
        'realistic': 0.2,
        'high': 20}
}


def b_band_merger_rate(luminosity,
                       population='NS-NS',
                       optimism='low'):

    r"""Model of Abadie et al (2010), Table III

    Compact binary merger rates as a linear function of a galaxies
    B-band luminosity.

    Parameters
    ----------
    luminosity : (ngal,) array-like
        The B-band luminosity of the galaxies to generate merger
        rates for, in units of solar luminosity.
    population : {'NS-NS', 'NS-BH', 'BH-BH'}
        Compact binary population to get rate for.
        'NS-NS' is neutron star - neutron star
        'NS-BH' is neutron star - black hole
        'BH-BH' is black hole - black hole
    optimism : {'low', 'realistic', 'high'}
        Optimism of predicted merger rates.
        For 'NS-NS' there is an extra option 'max'.

    Returns
    -------
    merger_rate : array_like
        Merger rates for the galaxies in units of year^-1

    Notes
    -----

    References
    ----------
    .. Abadie et al. 2010, Classical and Quantum Gravity,
        Volume 27, Issue 17, article id. 173001 (2010)
        https://arxiv.org/abs/1003.2480

    Examples
    --------
    >>> import numpy as np
    >>> from skypy.gravitational_wave import b_band_merger_rate

    Sample 100 luminosity values near absolute magnitude -20.5.

    >>> luminosities = 10.**(-0.4*(-20.5 + np.random.randn(100)))

    Generate merger rates for these luminosities.

    >>> rates = b_band_merger_rate(luminosities,
    ...                            population='NS-NS',
    ...                            optimism='low')

    """

    # Convert luminosity to units of L_10 defined in Abadie et. al. 2010
    L_10 = luminosity * constants.L_sun.to_value('erg/s') / (1e10 * 2.16e33)
    return abadie_table_III[population][optimism] * L_10 / units.year
