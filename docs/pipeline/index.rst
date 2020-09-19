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

The `~skypy.pipeline.Pipeline` class can be used to run SkyPy programmatically
from other code.

.. code-block:: python

    from skypy.pipeline import Pipeline

    # read the MCCL example pipeline
    driver = Pipeline.read('examples/mccl_galaxies.yml')

    # run the pipeline
    driver.execute()

    # access the results
    print(driver['blue_galaxies'])


Reference/API
=============

.. automodapi:: skypy.pipeline
.. automodapi:: skypy.pipeline.scripts.skypy
