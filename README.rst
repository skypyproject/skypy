===========================================
SkyPy: A package for modelling the Universe
===========================================

|Read the Docs| |GitHub| |Codecov| |Compatibility|

This package contains methods for modelling the Universe, galaxies and the
Milky Way. SkyPy simulates populations of astronomical objects, generating
random realisations of intrinsic and observed properties, with the
intention the simulations can then be compared to data as part of an inference
pipeline.

Currently, SkyPy implements the following modules:

* Galaxies_: morphology, luminosity and redshift distributions
* Pipelines_ to generate populations of astronomical objects

The `full list of features`_ can be found in the `SkyPy Documentation`_.

For more information on the people involved and how SkyPy is developed, please
visit the SkyPy Collaboration website: `http://skypyproject.org`_

.. _Galaxies: https://skypy.readthedocs.io/en/latest/galaxies.html
.. _Pipelines: https://skypy.readthedocs.io/en/latest/pipeline/index.html
.. _full list of features: https://skypy.readthedocs.io/en/latest/feature_list.html
.. _SkyPy Documentation: https://skypy.readthedocs.io/en/latest/
.. _http://skypyproject.org: http://skypyproject.org

Citation
--------

|JOSS| |Zenodo|

If you use SkyPy for work or research presented in a publication please follow
our `Citation Guidelines`_.

.. _Citation Guidelines: CITATION.rst


Installation
------------

|PyPI| |conda-forge|

SkyPy releases are distributed through PyPI_ and conda-forge_. Instructions for
installing SkyPy and its dependencies can be found in the Installation_
section of the documentation.


Examples
--------

SkyPy has a driver script that can run simulation pipelines from the command
line. The documentation contains a description of the Pipeline_ module and
Examples_ that demonstrate how to use it.

.. _PyPI: https://pypi.org/project/skypy/
.. _conda-forge: https://anaconda.org/conda-forge/skypy
.. _Installation: https://skypy.readthedocs.io/en/stable/install.html
.. _Pipeline: https://skypy.readthedocs.io/en/stable/pipeline/index.html
.. _Examples: https://skypy.readthedocs.io/en/stable/examples/index.html


Contributing
------------

We love contributions! 
SkyPy is open source,
built on open source, and we'd love to have you hang out in our community.

How to contribute
^^^^^^^^^^^^^^^^^

Whether you would like to contribute to SkyPy with your own piece of code or
helping develop a concrete feature in SkyPy:

1. Read through our `Discussions Page`_ to start a new conversation and share your
ideas or follow up an existing conversation on a particular feature.

2. Following the discussions, when you have a good idea of the specifics 
of the feature you wish to contribute, open an `Issue`_ describing the feature. 

3. Then follow the `Contributor Guidelines`_ to open a `Pull Request`_ to contribute
the code implementing the new feature.

For further information on how to contribute see our `Contributor Guidelines`_.
All communication relating to The SkyPy Project must meet the standards set out
in the `Code of Conduct`_.

.. _Issue: https://github.com/skypyproject/skypy/issues
.. _Pull Request: https://github.com/skypyproject/skypy/pulls
.. _Contributor Guidelines: https://skypy.readthedocs.io/en/latest/developer/contributing.html
.. _Code of Conduct: https://skypy.readthedocs.io/en/stable/project/code_of_conduct.html

Members vs External contributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SkyPy allows contributions from two types of contributor: *Members* and *External Contributors*.
These two categories are intended to allow contributions both from those who are willing and
able to commit to being part of the SkyPy community and actively involved in the steering of the project,
and those who wish to simply contribute code where a need has been identified.

1. SkyPy *Members* go through a simple onboarding process where their expertise and expected contributions
are discussed and defined. Members have access to internal communication channels, they are involved in
SkyPy decision making processes and attend quarterly meetings.
Members are listed as a separate tier in author lists for SkyPy publications,
with the classification of "Project Members" in the Zenodo DoI.

2. *External Contributors* are able to develop, discuss and commit code in the same way as *Members*,
but do not have the same responsibilities and opportunities for contributing to the guidance and management
of SkyPy as a project. *External Contributors* are listed as a separate tier in author lists for SkyPy publications,
with the classification of "Others" in the Zenodo version DoI.

Get in Touch
------------

You are welcome to talk about the SkyPy package and code using our
`Discussions Page`_. For any other questions about the project in general,
please get in touch with the `SkyPy Co-ordinators`_.

 .. _Discussions Page: https://github.com/skypyproject/skypy/discussions
 .. _SkyPy Co-ordinators: mailto:skypy-coordinators@googlegroups.com

.. |PyPI| image:: https://img.shields.io/pypi/v/skypy?label=PyPI&logo=pypi
    :target: https://pypi.python.org/pypi/skypy

.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/skypy?logo=conda-forge
    :target: https://anaconda.org/conda-forge/skypy

.. |Read the Docs| image:: https://img.shields.io/readthedocs/skypy/stable?label=Docs&logo=read%20the%20docs
    :target: https://skypy.readthedocs.io/en/stable

.. |GitHub| image:: https://github.com/skypyproject/skypy/workflows/Tests/badge.svg
    :target: https://github.com/skypyproject/skypy/actions

.. |Compatibility| image:: https://github.com/skypyproject/skypy/actions/workflows/compatibility.yaml/badge.svg
    :target: https://github.com/skypyproject/skypy/actions/workflows/compatibility.yaml

.. |Codecov| image:: https://codecov.io/gh/skypyproject/skypy/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/skypyproject/skypy

.. |Zenodo| image:: https://zenodo.org/badge/221432358.svg
    :target: https://zenodo.org/badge/latestdoi/221432358
    :alt: SkyPy Concept DOI

.. |JOSS| image:: https://joss.theoj.org/papers/d4fac0604318190d6627ab29b568a48d/status.svg
    :target: https://joss.theoj.org/papers/d4fac0604318190d6627ab29b568a48d
