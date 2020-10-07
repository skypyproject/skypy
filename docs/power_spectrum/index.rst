***************************************
Power Spectrum (`skypy.power_spectrum`)
***************************************
This module contains methods to model the matter power spectrum.

You can plot linear power spectrum using Eisenstein and Hu
fitting formula:

.. literalinclude:: examples/power_spectrum.yml
   :language: yaml


.. plot::
   :include-source: false
   :nofigs:
   :context: close-figs

    from skypy.pipeline import Pipeline
    pipeline = Pipeline.read('examples/power_spectrum.yml')
    pipeline.execute()


.. plot::
   :include-source: true
   :context: close-figs

    # Eisenstein and Hu power spectrum
    wavenumber = pipeline['k']
    power_EH = pipeline['power_spectrum']

    plt.loglog(wavenumber, power_EH, label='Wiggles')
    plt.xlabel(r'Wavenumber $(1/Mpc)$')
    plt.ylabel(r'Power spectrum $(Mpc^3)$')
    plt.title('Linear power spectrum')
    plt.legend()
    plt.show()



.. automodule:: skypy.power_spectrum
