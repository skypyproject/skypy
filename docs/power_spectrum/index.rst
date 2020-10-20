***************************************
Power Spectrum (`skypy.power_spectrum`)
***************************************
This module contains methods to model the matter power spectrum.

You can plot the linear power spectrum using CAMB or CLASS from SkyPy.

Here we plot the Eisenstein and
Hu fitting formula and the non-linear power spectrum, using the Smith
halofit model. For this we use a config file. We

.. literalinclude:: examples/power_spectrum.yml
   :language: yaml

.. plot::
   :include-source: false
   :context: close-figs

    import matplotlib.pyplot as plt
    from skypy.pipeline import Pipeline

    pipeline = Pipeline.read('examples/power_spectrum.yml')
    pipeline.execute()

    # Eisenstein and Hu power spectrum with and without wiggles
    k = pipeline['wavenumber']
    power_EH_w = pipeline['eisenstein_hu_wiggle']
    hf_Smith = pipeline['halofit']


    plt.loglog(k, power_EH_w, label='Eisenstein & Hu')
    plt.loglog(k, hf_Smith, '--', label='Halofit')
    plt.xlabel(r'Wavenumber $(1/Mpc)$')
    plt.ylabel(r'Power spectrum $(Mpc^3)$')
    plt.legend(frameon=False, loc='lower left');



.. automodule:: skypy.power_spectrum
