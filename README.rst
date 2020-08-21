===========================================
SkyPy: A package for modelling the Universe
===========================================

|Zenodo Badge| |Astropy Badge| |Test Status| |Coverage Status| |PyPI Status| |Anaconda Status| |Documentation Status|

This package contains methods for modelling the Universe, galaxies and the Milky
Way. Also included are methods for generating observed data.

* `Galaxy <https://skypy.readthedocs.io/en/latest/galaxy.html>`_ morphology, luminosity and redshift distributions
* `Halo <https://skypy.readthedocs.io/en/latest/halo/index.html>`_ and subhalo mass distributions
* `Gravitational Wave <https://skypy.readthedocs.io/en/latest/gravitational_wave/index.html>`_ binary merger rates
* `Power Spectra <https://skypy.readthedocs.io/en/latest/power_spectrum/index.html>`_ using CAMB and Halofit

SkyPy can also implement `Pipelines <https://skypy.readthedocs.io/en/latest/pipeline/index.html>`_
that generate populations of astronomical objects. The full list of features can
be found in the `SkyPy Documentation <https://skypy.readthedocs.io/en/latest/>`_.


Important links
---------------

* `Code of Conduct <https://github.com/skypyproject/skypy/blob/master/CODE_OF_CONDUCT.md>`_
* `Contributor Guidelines <https://github.com/skypyproject/skypy/blob/master/CONTRIBUTING.md>`_
* `Citation Guidelines <https://github.com/skypyproject/skypy/blob/master/CITATION>`_


How to install and run SkyPy
----------------------------

* Requirements:

  Make sure you have current version of ``astropy``, ``networkx``,
  ``numpy``, ``scipy`` and ``pyyaml``.

* Installation with pip:

  ``pip install skypy``

* Installation with conda:

  ``conda install -c conda-forge skypy``

* Running tests:

  ``pytest --pyargs skypy``

* Importing the package:

  You can use your favorite `python` shell (python, ipython, jupyter notebook),
  and import

  ``import skypy``

* Running examples:

  Use skypy to run one of the example pipelines and write the outputs to fits files.

  You could clone or download our repository or simply download the example directory.

  Move to the directory and type on your terminal

  ``skypy –config examples/herbel_galaxies.yaml –format fits``


Contributing
------------

We love contributions! SkyPy is open source,
built on open source, and we'd love to have you hang out in our community.

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
