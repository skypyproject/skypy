***************************
Pipeline (`skypy.pipeline`)
***************************

The `~skypy.pipeline` package contains the functionality to run a SkyPy
simulation from end to end. This is implemented in the `~skypy.pipeline.Pipeline`
class and can be called using the :ref:`skypy command line script <skypy-script>`.


.. _skypy-script:

Running ``skypy`` from the command line
=======================================

``skypy`` is a command line script that runs a pipeline of functions defined in
a config file to generate tables of objects and write them to file. For example,
you can use ``skypy`` to run one of the `example pipelines`_ and write the
outputs to fits files:

.. code-block:: bash

    $ skypy examples/mccl_galaxies.yml --format fits

Config files are written in YAML format and read using the
`~skypy.pipeline.load_skypy_yaml` funciton. Each entry in the config specifices
an arbitrary variable, but there are also some particular fields that SkyPy uses:

- `parameters` : Variables that can be modified at execution
- `cosmology` : The cosmology to be used by functions within the pipeline
- `tables` : A dictionary of tables names, each resolving to a dictionary of column names for that table

Every variable can be assigned a fixed value as parsed by pyyaml_.
However, variables and columns can also be evaluated as functions. Fuctions are
defined as tuples where the first entry is the fully qualified function name
tagged with and exclamation mark ``!`` and the second entry is either a list
of positional arguments or a dictionary of keyword arguments. Variables
and columns in the pipeline can also be referenced by their full name tagged
with a dollar sign ``$``. For example:

.. literalinclude:: examples/config.yml
   :language: yaml
   :caption:

.. plot::
  :include-source: false

    import matplotlib.pyplot as plt
    from skypy.pipeline import Pipeline

    pipeline = Pipeline.read('examples/config.yml')
    pipeline.execute()

    z = pipeline['galaxies']['redshift']
    dm = pipeline['galaxies']['distance_modulus']

    plt.scatter(z, dm, marker='+', color='k', label='galaxies')
    plt.legend()
    plt.xlabel('redshift')
    plt.ylabel('distance_modulus')

When executing a pipeline, all dependencies are tracked and resolved in order
using a Directed Acylic Graph implemented in networkx_.

.. _Example Pipelines: https://github.com/skypyproject/skypy/tree/master/examples
.. _pyyaml: https://pyyaml.org/
.. _networkx: https://networkx.github.io/


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

    import matplotlib.pyplot as plt
    from skypy.pipeline import Pipeline

    # read the example pipeline
    pipeline = Pipeline.read('examples/pipeline.yml')

    # run the pipeline as given
    pipeline.execute()

    # plot the results for the given parameters
    plt.hist(pipeline['galaxy-redshifts'], histtype='step', density=True,
             label='{:.2f}'.format(pipeline.parameters['median-redshift']))

    # change the median redshift parameter in a loop
    for z in [1.2, 1.4, 1.6, 1.8, 2.0]:

        # median redshift parameter
        parameters = {'median-redshift': z}

        # run pipeline with updated parameters
        pipeline.execute(parameters)

        # plot the new results
        plt.hist(pipeline['galaxy-redshifts'], histtype='step', density=True,
                 label='{:.2f}'.format(parameters['median-redshift']))

    # show plot labels
    plt.legend()
    plt.xlabel('redshift')


Lightcone Simulations
=====================

The `~skypy.pipeline.Lightcone` class can be used to simulate a cosmological
volume by subdiving it into a number of redshift slices and then running a
pipeline for each slice. A typical config file would be:

.. literalinclude:: examples/lightcone.yml
   :language: yaml
   :caption:

This simulates between redshift 0.8 and 1.2 in four slices. The slices are
determined by dividing the simulation volume into equally spaced comoving
distance slices. For each slice the lightcone defines three additional
parameters that can be referenced within the pipeline:

- `slice_z_min` : The minimum redshift of the slice
- `slice_z_max` : The maximum redshift of the slice
- `slice_z_mid` : The redshift of the comoving distance midpoint of the slice.

The redshifts of each galaxy in this simulated lightcone are shown below
colour-coded by the slice that they were sampled in.

.. plot::
  :include-source: false

    from astropy.cosmology import default_cosmology
    import matplotlib.pyplot as plt
    import numpy as np
    from skypy.pipeline import Lightcone, load_skypy_yaml

    # Run Lightcone
    config = load_skypy_yaml('examples/lightcone.yml')
    lightcone = Lightcone(config)
    lightcone.execute()

    # Lightcone outputs
    z = lightcone.tables['galaxies']['redshift']
    chi = default_cosmology.get().comoving_distance(z)
    ids = np.arange(len(z))

    # Redshift slices
    chi_min = default_cosmology.get().comoving_distance(lightcone.lightcone_config['z_min'])
    chi_max = default_cosmology.get().comoving_distance(lightcone.lightcone_config['z_max'])
    n_slice = lightcone.lightcone_config['n_slice']
    bounds = np.linspace(chi_min, chi_max, n_slice + 1)

    # Plotting
    colors = ['red', 'orange', 'green', 'blue']
    for i, c, l, h in zip(np.arange(n_slice), colors, bounds[:-1], bounds[1:]):
      mask = np.logical_and(chi > l, chi < h)
      plt.scatter(ids[mask], z[mask], marker='+', label=f'Slice {i+1}', color=c)
    plt.xlabel('Galaxy ID')
    plt.ylabel('Redshift')
    plt.legend()

Reference/API
=============

.. automodapi:: skypy.pipeline
