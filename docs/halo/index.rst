********************************
Dark Matter Halos (`skypy.halo`)
********************************

.. automodule:: skypy.halo

You can reproduce figure 2 in Sheth and Tormen 1999
and plot the collapse functions for different halo models. For this, you can use
a python script, for example.

.. plot::
   :include-source: true
   :nofigs:
   :context: close-figs

    import numpy as np
    from astropy.cosmology import Planck15
    from skypy.power_spectrum import eisenstein_hu
    from skypy.halo.mass import _sigma_squared

    # Power spectrum and amplitude of perturbations at redshift 0
    growth_0 = 1.0
    A_s, n_s = 2.1982e-09, 0.969453
    k = np.logspace(-3, 1, num=1000, base=10.0)
    mass = 10**np.arange(9.0, 15.0, 0.1)

    pk0 = eisenstein_hu(k, A_s, n_s, Planck15, kwmap=0.02, wiggle=True)
    sigma = np.sqrt(_sigma_squared(mass, k, pk0, growth_0, Planck15))

    # Collapse functions
    from skypy.halo import mass

    params_ellipsoidal = (0.3, 0.7, 0.3, 1.686)

    ST = mass.sheth_tormen_collapse_function(sigma)
    PS = mass.press_schechter_collapse_function(sigma)
    EM = mass.ellipsoidal_collapse_function(sigma, params=params_ellipsoidal)


.. plot::
   :include-source: false
   :context: close-figs

    import matplotlib.pyplot as plt

    delta_c = 1.69
    nu = np.square(delta_c / sigma)

    # plot different collapse functions
    plt.loglog(nu, ST, label='Sheth-Tormen')
    plt.loglog(nu, PS, label='Press-Schechter')
    plt.loglog(nu, EM, label='Ellipsoidal')

    # axes labels and title
    plt.xlabel(r'$\nu \equiv (\delta_c / \sigma)^2$')
    plt.ylabel(r'$f_c(\nu)$')
    plt.title('Collapse function')

    # show plot labels
    plt.legend()
    plt.show()


You can also sample halos using their mass function. For this, you can use a config
file and run the pipeline, for example.

.. literalinclude:: examples/halo.yml
   :language: yaml

.. plot::
   :include-source: false
   :context: close-figs

    from skypy.pipeline import Pipeline
    pipeline = Pipeline.read('examples/halo.yml')
    pipeline.execute()

    # Draw from different halo mass samplers
    halo_massST = pipeline['sheth-tormen']
    halo_massPS = pipeline['press-schechter']

    plt.hist(np.log10(halo_massST), histtype='step', label='Sheth-Tormen')
    plt.hist(np.log10(halo_massPS), histtype='step', label='Press-Schechter')

    # axis label and title
    plt.xlabel(r'$log(mass)$')
    plt.title('Halo sampler')

    # show plot labels
    plt.legend()
    plt.show()


Abundance Matching (`skypy.halo.abundance_matching`)
====================================================

.. automodule:: skypy.halo.abundance_matching


Mass (`skypy.halo.mass`)
========================

.. automodule:: skypy.halo.mass


Quenching (`skypy.halo.quenching`)
==================================

.. automodule:: skypy.halo.quenching
