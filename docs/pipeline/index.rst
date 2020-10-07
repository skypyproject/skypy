***************************
Pipeline (`skypy.pipeline`)
***************************


Running SkyPy
=============

The `~skypy.pipeline` package contains the functionality to run a SkyPy
simulation from end to end. This is implemented in the `~skypy.pipeline.Pipeline`
class and can be called using the :ref:`skypy command line script <skypy-script>`.


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


.. _skypy-script:

Running ``skypy`` from the command line
=======================================

``skypy`` is a command line script that runs a pipeline of functions defined in
a config file to generate tables of objects and write them to file.

Using ``skypy`` to run one of the example pipelines and write the outputs to
fits files:

.. code-block:: bash

    $ skypy examples/mccl_galaxies.yml --format fits


Config Files
------------

Config files are written in yaml format. The top level should contain the
fields ``cosmology`` and/or ``tables``. ``cosmology`` should contain a
dictionary configuring a function that returns an
``astropy.cosmology.Cosmology`` object. ``tables`` should contain a set of
nested dictionaries, first giving the names of each table, then the names of
each column within each table. Each column should contain a dictionary
configuring a function that returns an array-like object.

Each step in the pipeline is configured by a dictionary specifying:

- 'function' : the name of the function
- 'module' : the name of the the module to import 'function' from
- 'args' : a list of positional arguments (by value)
- 'kwargs' : a dictionary of keyword arguments
- 'requires' : a dictionary of keyword arguments

Note that 'kwargs' specifices keyword arguments by value, wheras 'requires'
specifices the names of previous steps in the pipeline and uses their return
values as keyword arguments.


Reference/API
=============

.. automodapi:: skypy.pipeline
