===========================================
SkyPy: A package for modelling the Universe
===========================================

|PyPI| |conda-forge| |Read the Docs| |GitHub| |Codecov| |Zenodo|

This package contains methods for modelling the Universe, galaxies and the Milky
Way. Also included are methods for generating observed data.

* Galaxy_ morphology, luminosity and redshift distributions
* Pipelines_ to generate populations of astronomical objects

The full list of features can be found in the `SkyPy Documentation`_.

If you use SkyPy for work or research presented in a publication please follow
our `Citation Guidelines`_.

.. _Galaxy: https://skypy.readthedocs.io/en/latest/galaxy.html
.. _Pipelines: https://skypy.readthedocs.io/en/latest/pipeline/index.html
.. _SkyPy Documentation: https://skypy.readthedocs.io/en/latest/
.. _Citation Guidelines: CITATION


Installation
------------

SkyPy releases are distributed through PyPI_ and conda-forge_. Instructions for
installing SkyPy and its dependencies can be found in the Installation_
section of the documentation.


Examples
--------

SkyPy also has a driver script that can run simulation pipelines from the
command line. The `skypyproject/examples`_ repository contains sample
configuration files that you can clone and run:

.. code:: bash

    $ git clone --depth 1 -b v$(skypy --version) https://github.com/skypyproject/examples.git
    $ skypy examples/mccl_galaxies.yml --format fits

.. _PyPI: https://pypi.org/project/skypy/
.. _conda-forge: https://anaconda.org/conda-forge/skypy
.. _Installation: https://skypy.readthedocs.io/en/stable/install.html
.. _skypyproject/examples: https://github.com/skypyproject/examples


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
