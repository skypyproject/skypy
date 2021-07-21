
r"""Binary merger rate module.
This module provides functions to calculate compact binary merger rates
for individual galaxies.

"""

import numpy as np
from astropy import constants, units


__all__ = [
    'b_band_merger_rate',
    'm_star_merger_rate',
    'm_star_sfr_merger_rate',
    'm_star_sfr_metallicity_merger_rate'
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

artale_tables = {
    'NS-NS': {
            'redshift': [0.1, 1.0, 2.0, 6.0],
            'alpha1': [1.038, 1.109, 1.050, 1.027],
            'alpha1_err': [0.001, 0.001, 0.001, 0.003],
            'alpha2': [-6.09, -6.214, -5.533, -5.029],
            'alpha2_err': [0.010, 0.006, 0.006, 0.021],
            'beta1': [0.800, 0.964, 0.977, 1.113],
            'beta1_err': [0.002, 0.001, 0.001, 0.003],
            'beta2': [0.323, 0.155, 0.068, -0.070],
            'beta2_err': [0.002, 0.001, 0.001, 0.002],
            'beta3': [-3.555, -4.819, -4.874, -5.764],
            'beta3_err': [0.018, 0.013, 0.011, 0.026],
            'gamma1': [0.701, 0.896, 1.018, 1.137],
            'gamma1_err': [0.002, 0.002, 0.002, 0.004],
            'gamma2': [0.356, 0.184, 0.048, -0.082],
            'gamma2_err': [0.002, 0.001, 0.001, 0.002],
            'gamma3': [0.411, 0.222, -0.103, -0.053],
            'gamma3_err': [0.005, 0.003, 0.002, 0.004],
            'gamma4': [-1.968, -3.795, -5.451, -6.104],
            'gamma4_err': [0.026, 0.019, 0.017, 0.037],
    },
    'NS-BH': {
            'redshift': [0.1, 1.0, 2.0, 6.0],
            'alpha1': [0.824, 0.873, 0.913, 0.965],
            'alpha1_err': [0.001, 0.001, 0.001, 0.002],
            'alpha2': [-4.731, -4.478, -4.401, -4.315],
            'alpha2_err': [0.008, 0.008, 0.007, 0.018],
            'beta1': [0.711, 0.813, 0.871, 0.985],
            'beta1_err': [0.002, 0.002, 0.002, 0.003],
            'beta2': [0.150, 0.064, 0.039, -0.017],
            'beta2_err': [0.002, 0.002, 0.001, 0.001],
            'beta3': [-3.536, -3.900, -4.019, -4.490],
            'beta3_err': [0.016, 0.018, 0.014, 0.024],
            'gamma1': [0.833, 1.074, 1.084, 0.978],
            'gamma1_err': [0.002, 0.002, 0.002, 0.003],
            'gamma2': [0.101, -0.058, -0.068, -0.013],
            'gamma2_err': [0.002, 0.002, 0.001, 0.002],
            'gamma3': [-0.461, -0.788, -0.535, 0.016],
            'gamma3_err': [0.004, 0.004, 0.003, 0.004],
            'gamma4': [-5.434, -7.733, -7.055, -4.386],
            'gamma4_err': [0.022, 0.023, 0.018, 0.034],
    },
    'BH-BH': {
            'redshift': [0.1, 1.0, 2.0, 6.0],
            'alpha1': [0.807, 0.813, 0.831, 0.933],
            'alpha1_err': [0.001, 0.001, 0.001, 0.004],
            'alpha2': [-4.310, -3.845, -3.600, -4.190],
            'alpha2_err': [0.006, 0.008, 0.008, 0.026],
            'beta1': [0.812, 0.840, 0.858, 1.053],
            'beta1_err': [0.001, 0.002, 0.002, 0.004],
            'beta2': [-0.006, -0.029, -0.026, -0.098],
            'beta2_err': [0.001, 0.002, 0.001, 0.002],
            'beta3': [-4.358, -4.109, -3.850, -5.213],
            'beta3_err': [0.013, 0.018, 0.015, 0.034],
            'gamma1': [0.921, 1.134, 1.135, 1.131],
            'gamma1_err': [0.001, 0.002, 0.002, 0.005],
            'gamma2': [-0.051, -0.172, -0.167, -0.137],
            'gamma2_err': [0.001, 0.002, 0.001, 0.002],
            'gamma3': [-0.404, -0.839, -0.681, -0.171],
            'gamma3_err': [0.003, 0.004, 0.003, 0.005],
            'gamma4': [-6.049, -8.338, -7.758, -6.321],
            'gamma4_err': [0.018, 0.024, 0.020, 0.047],
    }
}


def m_star_merger_rate(redshift,
                       m_star,
                       population):
    r"""Model of Artale et al (2020), equation (1) with parameters
    from Tables I, II and III.

    Compact binary merger rates as a log-log function of a galaxy's
    stellar mass.

    Parameters are redshift dependent, with linear interpolation
    between the simulated points of z={0.1, 1, 2, 6}.

    Parameters
    ----------
    redshift : (ngal,) array-like
        The redshifts of the galaxies to generate merger
        rates for.
    m_star : (ngal,) array-like
        The stellar mass of the galaxies to generate merger
        rates for, in units of stellar mass.
    population : {'NS-NS', 'NS-BH', 'BH-BH'}
        Compact binary population to get rate for.
        'NS-NS' is neutron star - neutron star
        'NS-BH' is neutron star - black hole
        'BH-BH' is black hole - black hole

    Returns
    -------
    merger_rate : array_like
        Merger rates for the galaxies in units of Gigayear^-1

    Notes
    -----

    References
    ----------
    .. Artale et al. 2020, MNRAS,
        Volume 491, Issue 3, p.3419-3434 (2020)
        https://arxiv.org/abs/1910.04890

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units
    >>> from skypy.gravitational_waves import m_star_merger_rate

    Sample 100 redshifts.

    >>> redshifts = np.random.uniform(0., 3., 100)

    Sample 100 stellar masses values near 10^9 solar masses.

    >>> stellar_masses = 10.**(9.0 + np.random.randn(100))

    Generate merger rates for these luminosities.

    >>> rates = m_star_merger_rate(redshifts,
    ...                            stellar_masses * units.Msun,
    ...                            population='NS-NS')

    """

    alpha1 = np.interp(redshift,
                       artale_tables[population]['redshift'],
                       artale_tables[population]['alpha1'])
    alpha2 = np.interp(redshift,
                       artale_tables[population]['redshift'],
                       artale_tables[population]['alpha2'])

    return _m_star_merger_rate(m_star, alpha1, alpha2)


def m_star_sfr_merger_rate(redshift,
                           m_star,
                           sfr,
                           population):
    r"""Model of Artale et al (2020), equation (2) with parameters
    from Tables I, II and III.

    Compact binary merger rates as a log-log function of a galaxy's
    stellar mass.

    Parameters are redshift dependent, with linear interpolation
    between the simulated points of z={0.1, 1, 2, 6}.

    Parameters
    ----------
    redshift : (ngal,) array-like
        The redshifts of the galaxies to generate merger
        rates for.
    m_star : (ngal,) array-like
        The stellar mass of the galaxies to generate merger
        rates for, in units of stellar mass.
    sfr : (ngal,) array-like
        The star formation rate of the galaxies to generate
        merger rates for, in units of stellar mass per year
    population : {'NS-NS', 'NS-BH', 'BH-BH'}
        Compact binary population to get rate for.
        'NS-NS' is neutron star - neutron star
        'NS-BH' is neutron star - black hole
        'BH-BH' is black hole - black hole

    Returns
    -------
    merger_rate : array_like
        Merger rates for the galaxies in units of Gigayear^-1

    Notes
    -----

    References
    ----------
    .. Artale et al. 2020, MNRAS,
        Volume 491, Issue 3, p.3419-3434 (2020)
        https://arxiv.org/abs/1910.04890

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units
    >>> from skypy.gravitational_waves import m_star_sfr_merger_rate

    Sample 100 redshifts.

    >>> redshifts = np.random.uniform(0., 3., 100)

    Sample 100 stellar masses values near 10^9 solar masses.

    >>> stellar_masses = 10.**(9.0 + np.random.randn(100))

    Sample 100 star formation rates.

    >>> sfrs = 10.**(np.random.randn(100))

    Generate merger rates for these luminosities.

    >>> rates = m_star_sfr_merger_rate(redshifts,
    ...                                stellar_masses * units.Msun,
    ...                                sfrs * units.Msun / units.year,
    ...                                population='NS-NS')

    """

    beta1 = np.interp(redshift,
                      artale_tables[population]['redshift'],
                      artale_tables[population]['beta1'])
    beta2 = np.interp(redshift,
                      artale_tables[population]['redshift'],
                      artale_tables[population]['beta2'])
    beta3 = np.interp(redshift,
                      artale_tables[population]['redshift'],
                      artale_tables[population]['beta3'])

    return _m_star_sfr_merger_rate(m_star, sfr, beta1, beta2, beta3)


def m_star_sfr_metallicity_merger_rate(redshift,
                                       m_star,
                                       sfr,
                                       Z,
                                       population):
    r"""Model of Artale et al (2020), equation (3) with parameters
    from Tables I, II and III.

    Compact binary merger rates as a log-log function of a galaxy's
    stellar mass.

    Parameters are redshift dependent, with linear interpolation
    between the simulated points of z={0.1, 1, 2, 6}.

    Parameters
    ----------
    redshift : (ngal,) array-like
        The redshifts of the galaxies to generate merger
        rates for.
    m_star : (ngal,) array-like
        The stellar mass of the galaxies to generate merger
        rates for, in units of stellar mass.
    sfr : (ngal,) array-like
        The star formation rate of the galaxies to generate
        merger rates for, in units of stellar mass per year
    Z : (ngal,) array-like
        The metallicity of the galaxies to generate merger
        rates for.
    population : {'NS-NS', 'NS-BH', 'BH-BH'}
        Compact binary population to get rate for.
        'NS-NS' is neutron star - neutron star
        'NS-BH' is neutron star - black hole
        'BH-BH' is black hole - black hole

    Returns
    -------
    merger_rate : array_like
        Merger rates for the galaxies in units of Gigayear^-1

    Notes
    -----

    References
    ----------
    .. Artale et al. 2020, MNRAS,
        Volume 491, Issue 3, p.3419-3434 (2020)
        https://arxiv.org/abs/1910.04890

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units
    >>> from skypy.gravitational_waves import m_star_sfr_metallicity_merger_rate

    Sample 100 redshifts.

    >>> redshifts = np.random.uniform(0., 3., 100)

    Sample 100 stellar masses values near 10^9 solar masses.

    >>> stellar_masses = 10.**(9.0 + np.random.randn(100))

    Sample 100 star formation rates.

    >>> sfrs = 10.**(np.random.randn(100))

    Sample 100 metallicities.

    >>> metallicities = 10.**(-2.0 + np.random.randn(100))

    Generate merger rates for these luminosities.

    >>> rates = m_star_sfr_metallicity_merger_rate(redshifts,
    ...                                            stellar_masses * units.Msun,
    ...                                            sfrs * units.Msun / units.year,
    ...                                            metallicities,
    ...                                            population='NS-NS')

    """

    gamma1 = np.interp(redshift,
                       artale_tables[population]['redshift'],
                       artale_tables[population]['gamma1'])
    gamma2 = np.interp(redshift,
                       artale_tables[population]['redshift'],
                       artale_tables[population]['gamma2'])
    gamma3 = np.interp(redshift,
                       artale_tables[population]['redshift'],
                       artale_tables[population]['gamma3'])
    gamma4 = np.interp(redshift,
                       artale_tables[population]['redshift'],
                       artale_tables[population]['gamma4'])

    return _m_star_sfr_metallicity_merger_rate(m_star, sfr, Z, gamma1, gamma2, gamma3, gamma4)


def _m_star_merger_rate(m_star,
                        alpha1,
                        alpha2):

    m_star = m_star.to(units.Msun).value

    n_gw = 10.**(alpha1 * np.log10(m_star) + alpha2)

    return n_gw / units.Gyr


def _m_star_sfr_merger_rate(m_star,
                            sfr,
                            beta1,
                            beta2,
                            beta3):

    m_star = m_star.to(units.Msun).value
    sfr = sfr.to(units.Msun / units.year).value

    n_gw = 10.**(beta1 * np.log10(m_star) + beta2 * np.log10(sfr) + beta3)

    return n_gw / units.Gyr


def _m_star_sfr_metallicity_merger_rate(m_star,
                                        sfr,
                                        Z,
                                        gamma1,
                                        gamma2,
                                        gamma3,
                                        gamma4):

    m_star = m_star.to(units.Msun).value
    sfr = sfr.to(units.Msun / units.year).value

    n_gw = 10.**(gamma1 * np.log10(m_star) + gamma2 * np.log10(sfr) + gamma3 * np.log10(Z) + gamma4)

    return n_gw / units.Gyr


def b_band_merger_rate(luminosity,
                       population='NS-NS',
                       optimism='low'):

    r"""Model of Abadie et al (2010), Table III

    Compact binary merger rates as a linear function of a galaxy's
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
    >>> from skypy.gravitational_waves import b_band_merger_rate

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
