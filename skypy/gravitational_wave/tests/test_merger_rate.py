import numpy as np
from astropy import constants, units
from skypy.gravitational_wave import b_band_merger_rate
from skypy.gravitational_wave.merger_rate import abadie_table_III


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
