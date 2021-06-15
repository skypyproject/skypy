############
Installation
############

This page outlines how to install one of the officially distributed SkyPy
releases and its dependencies, or install and test the latest development
version.

From PyPI
---------

All  SkyPy releases are distributed through the Python Package Index (PyPI_).
To install the latest version use pip_:

.. code:: console

    $ pip install skypy

From conda-forge
----------------

All SkyPy releases are also distributed for conda_ through the `conda-forge`_
channel. To install the latest version for your active conda environment:

.. code:: console

    $ conda install -c conda-forge skypy

From GitHub
-----------

The latest development version of SkyPy can be found on the main branch of
the `skypyproject/skypy`_ GitHub repository. This and any other branch or tag
can be installed directly from GitHub using a recent version of pip:

.. code:: console

    $ pip install skypy@git+https://github.com/skypyproject/skypy.git@main

Dependencies
------------

SkyPy is compatble with Python versions 3.6 or later on Ubuntu, macOS and
Windows operating systems. It has the following core dependencies:

- `astropy <https://www.astropy.org/>`__
- `networkx <https://networkx.github.io/>`_
- `numpy <https://numpy.org/>`_
- `pyyaml <https://pyyaml.org/>`_
- `scipy <https://www.scipy.org/>`_

Installing using pip or conda will automatically install or update these core
dependencies if necessary. SkyPy also has a number of optional dependencies
that enable additional features:

- `h5py <https://www.h5py.org/>`_
- `speclite <https://speclite.readthedocs.io/>`_

To install SkyPy with all optional dependencies using pip:

.. code:: console

    $ pip install skypy[all]

Testing
-------

Once installed, you should be able to import the `skypy` module in python:

.. code:: python

    >>> import skypy

You should also be able to check the installed version number using the `skypy`
command line script:

.. code:: console

    $ skypy --version

You may also want to run the unit tests, for example if you have installed the
development version or you use an unsupported operating system. The unit tests
have the following additional dependencies:

- `pytest-astropy <https://github.com/astropy/pytest-astropy>`_
- `pytest-rerunfailures <https://github.com/pytest-dev/pytest-rerunfailures>`_

The test dependencies can be installed using pip:

.. code:: console

    $ pip install skypy[test]

and the unit tests can then be run using pytest_:

.. code:: console

    $ pytest --pyargs skypy

.. _PyPI: https://pypi.org/project/skypy/
.. _pip: https://pip.pypa.io/
.. _conda: https://docs.conda.io/
.. _conda-forge: https://anaconda.org/conda-forge/skypy
.. _skypyproject/skypy: https://github.com/skypyproject/skypy
.. _pytest: https://docs.pytest.org/
