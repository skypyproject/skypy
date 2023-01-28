***************************************
Power Spectrum (`skypy.power_spectrum`)
***************************************
This module contains methods to model the matter power spectrum.

SkyPy provides wrappers to a number of external codes for calculating the
matter power spectrum, including `~skypy.power_spectrum.camb` and
`~skypy.power_spectrum.classy`. Here we demonstrate calculating the linear
matter power spectrum using `~skypy.power_spectrum.eisenstein_hu` and the
non-linear corrections using `~skypy.power_spectrum.halofit_smith`:

.. literalinclude:: examples/power_spectrum.yml
   :language: yaml

.. plot::
   :include-source: false
   :context: close-figs

    import matplotlib.pyplot as plt
    from skypy.pipeline import Pipeline

    pipeline = Pipeline.read('examples/power_spectrum.yml')
    pipeline.execute()

    # Eisenstein and Hu power spectrum and Halofit matter power spectra
    k = pipeline['wavenumber']
    power_EH_w = pipeline['eisenstein_hu_wiggle']
    hf_Smith = pipeline['halofit']

    plt.loglog(k, power_EH_w, label='Eisenstein & Hu')
    plt.loglog(k, hf_Smith, '--', label='Halofit')
    plt.xlabel(r'Wavenumber $(1/Mpc)$')
    plt.ylabel(r'Power spectrum $(Mpc^3)$')
    plt.legend(frameon=False, loc='lower left');



.. automodule:: skypy.power_spectrum
