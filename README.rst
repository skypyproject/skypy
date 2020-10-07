===========================================
SkyPy: A package for modelling the Universe
===========================================

|Zenodo Badge| |Astropy Badge| |Test Status| |Coverage Status| |PyPI Status| |Anaconda Status| |Documentation Status|

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

You can test your SkyPy intallation using pytest_:

.. code:: bash

    $ pytest --pyargs skypy

The SkyPy library can then be imported from python:

.. code:: python

    >>> import skypy
    >>> help(skypy)

SkyPy has a number of optional dependencies which can be installed separately.
One of these is `skypy-data`_ which contains data such as photometric bandpasses
required for some calculations in SkyPy. This can be installed with:

.. code:: bash
    
    pip install skypy-data@https://github.com/skypyproject/skypy-data/archive/master.tar.gz

SkyPy also has a driver script that can run simulation pipelines from the
command line. The `skypyproject/examples`_ repository contains sample
configuration files that you can clone and run:

.. code:: bash

    git clone --depth 1 -b v$(skypy --version) https://github.com/skypyproject/examples.git
    skypy examples/mccl_galaxies.yml --format fits

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

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
SkyPy based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.

.. |Zenodo Badge| image:: https://zenodo.org/badge/221432358.svg
   :target: https://zenodo.org/badge/latestdoi/221432358
   :alt: DOI of Latest SkyPy Release

.. |Astropy Badge| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

.. |Test Status| image:: https://github.com/skypyproject/skypy/workflows/Tests/badge.svg
    :target: https://github.com/skypyproject/skypy/actions
    :alt: SkyPy's Test Status

.. |Coverage Status| image:: https://codecov.io/gh/skypyproject/skypy/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/skypyproject/skypy
    :alt: SkyPy's Coverage Status

.. |PyPI Status| image:: https://img.shields.io/pypi/v/skypy.svg
    :target: https://pypi.python.org/pypi/skypy
    :alt: SkyPy's PyPI Status

.. |Anaconda Status| image:: https://anaconda.org/conda-forge/skypy/badges/version.svg
    :target: https://anaconda.org/conda-forge/skypy
    :alt: SkyPy's Anaconda Status

.. |Documentation Status| image:: https://readthedocs.org/projects/githubapps/badge/?version=latest
    :target: https://skypy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
