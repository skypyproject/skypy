***************************************
Power Spectrum (`skypy.power_spectrum`)
***************************************
This module contains methods to model the matter power spectrum.

You can plot linear power spectrum using CAMB from SkyPy
and different halofit models:

.. plot::
   :include-source: true
   :context: close-figs

    import numpy as np
    from astropy.cosmology import Planck15
    import matplotlib.pyplot as plt
    from skypy.power_spectrum import camb
    from skypy.power_spectrum import halofit_smith, halofit_takahashi, halofit_bird


    z0 = 1.0e-6
    k = np.logspace(-3.0, 0.0, 100)
    # Camb linear
    pc = camb(k, z0, Planck15, A_s=2.e-9, n_s=0.965)

    # Halofit models
    hf_Smith = halofit_smith(k, z0, pc, Planck15)
    hf_Tak = halofit_takahashi(k, z0, pc, Planck15)
    hf_Bird = halofit_bird(k, z0, pc, Planck15)

    plt.loglog(k, pc, label='Camb')
    plt.loglog(k, hf_Smith, label='Smith', lw=1)
    plt.loglog(k, hf_Tak, '--', label='Takahashi')
    plt.loglog(k, hf_Bird, ':', label='Bird')
    plt.xlabel(r'Wavenumber $(1/Mpc)$')
    plt.ylabel(r'Power spectrum $(Mpc^3)$')
    plt.legend(frameon=False)


You can also plot the Eisenstein and Hu fitting formula using a config file:

.. literalinclude:: examples/power_spectrum.yml
   :language: yaml

.. plot::
   :include-source: false
   :context: close-figs

    from skypy.pipeline import Pipeline
    pipeline = Pipeline.read('examples/power_spectrum.yml')
    pipeline.execute()

    # Eisenstein and Hu power spectrum with and without wiggles
    k = pipeline['wavenumber']
    power_EH_w = pipeline['eisenstein_hu_wiggle']
    power_EH_nw = pipeline['eisenstein_hu_nowiggle']


    plt.loglog(k, power_EH_w, label='Wiggles')
    plt.loglog(k, power_EH_nw, '--', label='Wiggles', lw=1)
    plt.xlabel(r'Wavenumber $(1/Mpc)$')
    plt.ylabel(r'Power spectrum $(Mpc^3)$')
    plt.title('Linear power spectrum')
    plt.legend(frameon=False);



.. automodule:: skypy.power_spectrum
