import numpy as np
from astropy import constants, units
from skypy.gravitational_waves import \
                                        b_band_merger_rate, \
                                        m_star_merger_rate, \
                                        m_star_sfr_merger_rate, \
                                        m_star_sfr_metallicity_merger_rate
from skypy.gravitational_waves.merger_rate import abadie_table_III, artale_tables


def test_abadie_rates():

    # Check the number of merger rates returned
    luminosity = 10.**(-0.4*(-20.5 + np.random.randn(1000)))
    rates = b_band_merger_rate(luminosity, population='NS-NS', optimism='low')
    assert len(rates) == len(luminosity)

    # Test that a luminosity of L_10 (in units of solar luminosity) returns a
    # rate that matches the value in Abadie Table III
    L_10 = 1e10 * 2.16e33 / constants.L_sun.to_value('erg/s')
    L_B_rate = b_band_merger_rate(L_10, population='NS-NS', optimism='low')
    table_value = abadie_table_III['NS-NS']['low']
    assert np.isclose(L_B_rate.to_value(1/units.year), table_value, rtol=1e-5)


def test_artale_rates():

    # Check the number of merger rates returned
    redshifts = np.random.uniform(0., 3., 100)
    stellar_masses = 10.**(9.0 + np.random.randn(100))
    rates = m_star_merger_rate(redshifts, stellar_masses * units.Msun, population='NS-NS')
    assert len(rates) == len(stellar_masses)

    # Test that a luminosity of M_sol returns a
    # rate that matches the value in Artale Table I
    m_sol = constants.M_sun.to_value('M_sun')
    sol_rate = m_star_merger_rate(0.0, m_sol * units.Msun, population='NS-NS')
    alpha1 = artale_tables['NS-NS']['alpha1'][0]
    alpha2 = artale_tables['NS-NS']['alpha2'][0]
    table_value = 10.**(alpha1 * np.log10(m_sol) + alpha2)
    assert np.isclose(sol_rate.to_value(1/units.Gyr), table_value, rtol=1e-5)

    # Test that a luminosity of M_sol and SFR of 1 returns a
    # rate that matches the value in Artale Table I
    m_sol = constants.M_sun.to_value('M_sun')
    sfr_1 = 1.
    sol_rate = m_star_sfr_merger_rate(0.0, m_sol * units.Msun,
                                      sfr_1 * units.Msun / units.year,
                                      population='NS-NS')
    beta1 = artale_tables['NS-NS']['beta1'][0]
    beta2 = artale_tables['NS-NS']['beta2'][0]
    beta3 = artale_tables['NS-NS']['beta3'][0]
    table_value = 10.**(beta1 * np.log10(m_sol) + beta2 * np.log10(sfr_1) + beta3)
    assert np.isclose(sol_rate.to_value(1/units.Gyr), table_value, rtol=1e-5)

    # Test that a luminosity of M_sol, SFR of 1 and Z of 0.012 returns a
    # rate that matches the value in Artale Table I
    m_sol = constants.M_sun.to_value('M_sun')
    sfr_1 = 1.
    Z_sol = 0.012
    sol_rate = m_star_sfr_metallicity_merger_rate(0.0, m_sol * units.Msun,
                                                  sfr_1 * units.Msun / units.year,
                                                  Z_sol,
                                                  population='NS-NS')
    gamma1 = artale_tables['NS-NS']['gamma1'][0]
    gamma2 = artale_tables['NS-NS']['gamma2'][0]
    gamma3 = artale_tables['NS-NS']['gamma3'][0]
    gamma4 = artale_tables['NS-NS']['gamma4'][0]
    table_value = 10.**(gamma1 * np.log10(m_sol) +
                        gamma2 * np.log10(sfr_1) +
                        gamma3 * np.log10(Z_sol) + gamma4)
    assert np.isclose(sol_rate.to_value(1/units.Gyr), table_value, rtol=1e-5)
