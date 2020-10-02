********************************
Dark Matter Halos (`skypy.halo`)
********************************

.. automodule:: skypy.halo

Halo mass function
==================
You can reproduce figure 2 in Sheth and Tormen 1999. For that, you can execute the
pipeline

.. literalinclude:: examples/halo.yml
   :language: yaml
   :caption:

And plot the collapse functions for different halo models:

.. plot::

    from skypy.pipeline import Pipeline

    # read the example pipeline
    pipeline = Pipeline.read('examples/pipeline.yml')

    # run the pipeline as given
    pipeline.execute()

    # Choose the halo models
    ST = pipeline['sheth-tormen-collapse-function']
    PS = pipeline['press-schechter-collapse-function']
    EM = pipeline['ellipsoidal-collapse-function']

    # plot different collapse functions
    plt.loglog(pipeline['nu'], ST, label='Sheth-Tormen')
    plt.loglog(pipeline['nu'], PS, label='Press-Schechter')
    plt.loglog(pipeline['nu'], EM, label='Ellipsoidal')

    # axes labels
    plt.xlabel(r'$\nu \equiv (\delta_c / \sigma)^2$')
    plt.ylabel(r'$f_c(\nu)$')
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
