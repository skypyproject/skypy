*****************
Running ``SkyPy``
*****************

As well as being imported within a python script or environment, the pipeline
driver :ref:`pipeline-driver-docs` can be used to run ``SkyPy`` as a standalone
executable.

We make available a number of yaml configuration files for running ``SkyPy`` in
this way. These can be found in the `examples directory of the repository
<https://github.com/skypyproject/skypy/tree/master/examples>`_.

These example configuration scripts can be run using the skypy command::

  skypy -c examples/herbel_galaxies.yaml -f fits

Here, the -c switch occurs immediately before the required configuration file,
and the -f switch occurs immediately before the requested file output format.

Running this example should generate two fits files in the same directory,
containing sampled redshifts and magnitudes for red and blue galaxies.