************************************************
Gravitational Waves (`skypy.gravitational_wave`)
************************************************

SkyPy provides methods to model physical processes involving gravitational waves.
Here we demonstrate calculating the merger rates for three
different types of compact binary mergers.
For this we use `~skypy.gravitational_wave.b_band_merger_rate` and
the configuration file provided in the `~skypy.examples` directory:


.. plot::
   :include-source: false

    from skypy.pipeline import Pipeline
    pipeline = Pipeline.read('examples/merger_rates.yml')
    pipeline.execute()
    merger = pipeline['merger_rates']

    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(merger['ns_ns_rate'], histtype='step', bins=np.logspace(-4,2,50), label='NS-NS')
    plt.hist(merger['ns_bh_rate'], histtype='step', bins=np.logspace(-4,2,50), label='NS-BH')
    plt.hist(merger['bh_bh_rate'], histtype='step', bins=np.logspace(-4,2,50), label='BH-BH')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Merger rate, $R\, [year^{-1}$]')

    plt.show()


Note: ‘NS-NS’ means neutron star - neutron star, ‘NS-BH’ is neutron star
- black hole and ‘BH-BH’ is black hole - black hole.

.. automodule:: skypy.gravitational_wave
