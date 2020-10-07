***************************************
Power Spectrum (`skypy.power_spectrum`)
***************************************
This module contains methods to model the matter power spectrum.

You can plot linear power spectrum using Eisenstein and Hu
fitting formula:

.. literalinclude:: examples/power_spectrum.yml
   :language: yaml


.. plot::
   :include-source: true
   :nofigs:
   :context: close-figs

    from skypy.pipeline import Pipeline
    pipeline = Pipeline.read('examples/power_spectrum.yml')
    pipeline.execute()

    # Eisenstein and Hu power spectrum with and without wiggles
    wavenumber = pipeline['k']
    power_EH_w = pipeline['eisenstein_hu_wiggle']
    power_EH_nw = pipeline['eisenstein_hu_nowiggle']


.. plot::
   :include-source: false
   :context: close-figs

    import matplotlib.pyplot as plt

    plt.loglog(wavenumber, power_EH_w, label='Wiggles')
    plt.loglog(wavenumber, power_EH_nw, '--', label='Wiggles', lw=1)
    plt.xlabel(r'Wavenumber $(1/Mpc)$')
    plt.ylabel(r'Power spectrum $(Mpc^3)$')
    plt.title('Linear power spectrum')
    plt.legend()
    plt.show()


.. automodule:: skypy.power_spectrum
