import numpy as np
from astropy import units
import skypy.galaxy.luminosity as lum
import skypy.gravitational_wave.merger_rate as merg


def test_abadie_rates():

    luminosities = lum.herbel_luminosities(1.0, -1.3, -0.9408582,
                                           -20.40492365, size=1000)

    abIII_rates = merg.b_band_merger_rate(luminosities,
                                          population='NS-NS',
                                          optimism='low')

    L_B = 1.772222222222222e10
    L_B_rate = merg.b_band_merger_rate(L_B,
                                       population='NS-NS',
                                       optimism='low')

    assert len(abIII_rates) == len(luminosities)
    assert np.isclose(L_B_rate.to(1. / units.year).value, 0.6, rtol=1e-1)
