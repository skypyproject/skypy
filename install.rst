************
Installation
************

.. _astropy-main-req:

Requirements
============

``SkyPy`` has the following strict requirements:

- `Python <https://www.python.org/>`_ |minimum_python_version| or later

- `Numpy`_ |minimum_numpy_version| or later

- `astropy`_ |minimum_numpy_version| or later

``SkyPy`` also depends on other packages for optional features:

- `scipy`_ |minimum_scipy_version| or later: To power a variety of features
  in several modules.

.. - `matplotlib <https://matplotlib.org/>`_ 2.0 or later: To provide plotting
  functionality that `astropy.visualization` enhances.

.. - `setuptools <https://setuptools.readthedocs.io>`_: Used for discovery of
  entry points which are used to insert fitters into `astropy.modeling.fitting`.

The following packages can optionally be used when testing:

- `tox <https://tox.readthedocs.io/en/latest/>`_: Used to automate testing
  and documentation builds.

Installing ``SkyPy``
======================

If you are new to Python and/or do not have familiarity with `Python virtual
environments <https://docs.python.org/3/tutorial/venv.html>`_, then we recommend
starting by installing the `Anaconda Distribution
<https://www.anaconda.com/distribution/>`_. This works on all platforms (linux,
Mac, Windows) and installs a full-featured scientific Python in a user directory
without requiring root permissions.

Using pip
---------

.. warning::

    Users of the Anaconda Python distribution should follow the instructions
    for :ref:`anaconda_install`.

To install ``SkyPy`` with `pip <https://pip.pypa.io>`__, run::

    pip install skypy

If you want to make sure none of your existing dependencies get upgraded, you
can also do::

    pip install skypy --no-deps

..On the other hand, if you want to install ``SkyPy`` along with all of the
.. available optional dependencies, you can do::

..    pip install astropy[all]

In most cases, this will install a pre-compiled version (called a *wheel*) of
SkyPy, but if you are using a very recent version of Python, if a new version
of SkyPy has just been released, or if you are building SkyPy for a platform
that is not common, SkyPy will be installed from a source file. Note that in
this case you will need a C compiler (e.g., ``gcc`` or ``clang``) to be installed
(see `Building from source`_ below) for the installation to succeed.

If you get a ``PermissionError`` this means that you do not have the required
administrative access to install new packages to your Python installation. In
this case you may consider using the ``--user`` option to install the package
into your home directory. You can read more about how to do this in the `pip
documentation <https://pip.pypa.io/en/stable/user_guide/#user-installs>`_.

Do **not** install ``SkyPy`` or other third-party packages using ``sudo``
unless you are fully aware of the risks.

.. .. _anaconda_install:

.. Using Conda
.. -----------

.. To install ``SkyPy`` using conda run::

..     conda install skypy

.. ``SkyPy`` is installed by default with the `Anaconda Distribution
.. <https://www.anaconda.com/distribution/>`_. To update to the latest version run::

..     conda update skypy

.. There may be a delay of a day or two between when a new version of ``SkyPy``
.. is released and when a package is available for conda. You can check
.. for the list of available versions with ``conda search skypy``.

.. It is highly recommended that you install all of the optional dependencies with::

..     conda install -c astropy -c defaults \
..       scipy h5py beautifulsoup4 html5lib bleach pyyaml pandas sortedcontainers \
..       pytz matplotlib setuptools mpmath bottleneck jplephem asdf

.. To also be able to run tests (see below) and support :ref:`builddocs` use the
.. following. We use ``pip`` for these packages to ensure getting the latest
.. releases which are compatible with the latest ``pytest`` and ``sphinx`` releases::

..     pip install pytest-astropy sphinx-astropy

.. .. warning::

..     Attempting to use `pip <https://pip.pypa.io>`__ to upgrade your installation
..     of ``SkyPy`` itself may result in a corrupted installation.

.. _testing_installed_astropy:

Testing an Installed ``SkyPy``
--------------------------------

The easiest way to test if your installed version of ``SkyPy`` is running
correctly is to use the :ref:`skypy.test()` function::

    import skypy
    skypy.test()

The tests should run and print out any failures, which you can report at
the `SkyPy issue tracker <https://github.com/skypyproject/skypy/issues>`_.

This way of running the tests may not work if you do it in the ``SkyPy`` source
distribution. See :ref:`sourcebuildtest` for how to run the tests from the
source code directory, or :ref:`running-tests` for more details.

.. Building from Source
.. ====================

Building Documentation
----------------------

.. note::

    Building the documentation is in general not necessary unless you are
    writing new documentation or do not have internet access, because
    the latest (and archive) versions of SkyPy's documentation should
    be available at `docs.astropy.org <http://readthedocs.org/projects/skypy>`_ .

Dependencies
^^^^^^^^^^^^

Building the documentation requires the ``SkyPy`` source code and some
additional packages. The easiest way to build the documentation is to use `tox
<https://tox.readthedocs.io/en/latest/>`_ as detailed in
:ref:`skypy-doc-building`. If you are happy to do this, you can skip the rest
of this section.

.. _skypy-doc-building:

Building
^^^^^^^^

There are two ways to build the Astropy documentation. The easiest way is to
execute the following tox command (from the ``SkyPy`` source directory)::

    tox -e build_docs

If you do this, you do not need to install any of the documentation dependencies
as this will be done automatically. The documentation will be built in the
``docs/_build/html`` directory, and can be read by pointing a web browser to
``docs/_build/html/index.html``.

Alternatively, you can do::

    cd docs
    make html

And the documentation will be generated in the same location. Note that
this uses the installed version of astropy, so if you want to make sure
the current repository version is used, you will need to install it with
e.g.::

    pip install -e .[docs]

before changing to the ``docs`` directory.

In the second way, LaTeX documentation can be generated by using the command::

    make latex

The LaTeX file ``skypy.tex`` will be created in the ``docs/_build/latex``
directory, and can be compiled using ``pdflatex``.