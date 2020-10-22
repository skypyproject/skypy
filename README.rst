===========================================
SkyPy: A package for modelling the Universe
===========================================

|PyPI| |conda-forge| |Read the Docs| |GitHub| |Codecov| |Zenodo|

This package contains methods for modelling the Universe, galaxies and the Milky
Way. Also included are methods for generating observed data.

* Galaxy_ morphology, luminosity and redshift distributions
* Halo_ and subhalo mass distributions
* `Gravitational Wave`_ binary merger rates
* `Power Spectra`_ using CAMB and Halofit
* Pipelines_ to generate populations of astronomical objects

The full list of features can be found in the `SkyPy Documentation`_.

If you use SkyPy for work or research presented in a publication please follow
our `Citation Guidelines`_.

.. _Galaxy: https://skypy.readthedocs.io/en/latest/galaxy.html
.. _Halo: https://skypy.readthedocs.io/en/latest/halo/index.html
.. _Gravitational Wave: https://skypy.readthedocs.io/en/latest/gravitational_wave/index.html
.. _Power Spectra: https://skypy.readthedocs.io/en/latest/power_spectrum/index.html
.. _Pipelines: https://skypy.readthedocs.io/en/latest/pipeline/index.html
.. _SkyPy Documentation: https://skypy.readthedocs.io/en/latest/
.. _Citation Guidelines: CITATION


Getting Started
---------------

SkyPy is distributed through PyPI_ and conda-forge_. To install SkyPy and its
dependencies_ using pip_:

.. code:: bash

    $ pip install skypy

To install using conda_:

.. code:: bash

    $ conda install -c conda-forge skypy

The SkyPy library can then be imported from python:

.. code:: python

    >>> import skypy

SkyPy has a number of optional dependencies which can be installed separately.
One of these is `skypy-data`_ which contains data such as photometric bandpasses
required for some calculations in SkyPy. This can be installed with:

.. code:: bash

    $ pip install skypy-data@https://github.com/skypyproject/skypy-data/archive/master.tar.gz

SkyPy also has a driver script that can run simulation pipelines from the
command line. The `skypyproject/examples`_ repository contains sample
configuration files that you can clone and run:

.. code:: bash

    $ git clone --depth 1 -b v$(skypy --version) https://github.com/skypyproject/examples.git
    $ skypy examples/mccl_galaxies.yml --format fits

.. _PyPI: https://pypi.org/project/skypy/
.. _conda-forge: https://anaconda.org/conda-forge/skypy
.. _dependencies: setup.cfg
.. _pip: https://pip.pypa.io/en/stable/
.. _conda: https://docs.conda.io/en/latest/
.. _pytest: https://docs.pytest.org/en/stable/
.. _skypyproject/examples: https://github.com/skypyproject/examples
.. _skypy-data: https://github.com/skypyproject/skypy-data


Contributing
------------

We love contributions! SkyPy is open source,
built on open source, and we'd love to have you hang out in our community.
For information on how to contribute see our `Contributor Guidelines`_.
All communication relating to The SkyPy Project must meet the standards set out
in the `Code of Conduct`_.

.. _Contributor Guidelines: CONTRIBUTING.md
.. _Code of Conduct: CODE_OF_CONDUCT.md

.. |PyPI| image:: https://img.shields.io/pypi/v/skypy?label=PyPI&logo=pypi
    :target: https://pypi.python.org/pypi/skypy

.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/skypy?logo=conda-forge
    :target: https://anaconda.org/conda-forge/skypy

.. |Read the Docs| image:: https://img.shields.io/readthedocs/skypy/stable?label=Docs&logo=read%20the%20docs
    :target: https://skypy.readthedocs.io/en/stable

.. |GitHub| image:: https://github.com/skypyproject/skypy/workflows/Tests/badge.svg
    :target: https://github.com/skypyproject/skypy/actions

.. |Codecov| image:: https://codecov.io/gh/skypyproject/skypy/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/skypyproject/skypy

.. |Zenodo| image:: https://zenodo.org/badge/221432358.svg
    :target: https://zenodo.org/badge/latestdoi/221432358
    :alt: SkyPy Concept DOI
