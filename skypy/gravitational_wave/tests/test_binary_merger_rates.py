import skypy.galaxy.luminosity as lum
import skypy.gravitational_waves.merger_rates as merg

def test_abadie_rates():

  luminosities = lum.herbel_luminosities(1.0, -1.3, -0.9408582,
                                         -20.40492365, size=100)

  abIII_rates = merg.abadie_tableIII_merger_rates(luminosities,
                                                  population='NS-NS',
                                                  optimism='low')

  assert len(abIII_rates)==len(luminosities)