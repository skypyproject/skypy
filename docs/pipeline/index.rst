***************************
Pipeline (`skypy.pipeline`)
***************************


Running SkyPy
=============

The `~skypy.pipeline` package contains the functionality to run a SkyPy
simulation from end to end. This is implemented in the `~skypy.pipeline.Pipeline`
class and can be called using the command line script
`~skypy.pipeline.scripts.skypy`.


Using a pipeline from other code
================================

SkyPy pipelines can be executed programmatically from other code. Consider the
following example configuration:

.. literalinclude:: examples/pipeline.yml
   :language: yaml
   :caption:

The `~skypy.pipeline.Pipeline` class can be used to load the configuration file
and run the resulting pipeline. If the configuration defines a `parameters`
section, the definition can be accessed and individual parameter values can be
changed for individual executions of the pipeline:

.. plot::

    from skypy.pipeline import Pipeline

    # read the example pipeline
    pipeline = Pipeline.read('examples/pipeline.yml')

    # run the pipeline as given
    pipeline.execute()

    # get the default parameters
    parameters = pipeline.parameters

    # plot the result from the default parameters
    import matplotlib.pyplot as plt
    plt.hist(pipeline['galaxy-redshifts'], histtype='step',
             label='{:.2f}'.format(parameters['median-redshift']))

    # change the parameters in a loop
    for i in range(5):

        # tweak the median redshift parameter
        parameters['median-redshift'] += 0.2

        # run pipeline with updated parameters
        pipeline.execute(parameters)

        # plot the new results
        plt.hist(pipeline['galaxy-redshifts'], histtype='step',
                 label='{:.2f}'.format(parameters['median-redshift']))

    # show plot labels
    plt.legend()


Reference/API
=============

.. automodapi:: skypy.pipeline
.. automodapi:: skypy.pipeline.scripts.skypy
