###################
Configuration files
###################

This page outlines how to construct configuration files to run your own routines
with `~skypy.pipeline.Pipeline`.

Remember that SkyPy not only is a library of astronomical models but a flexible
tool with infrastructure for you to use with your own
functions and pipelines!

YAML syntax
-----------
YAML_ is a file format designed to be readable by both computers and humans.
Fundamentally, a file written in YAML consists of a set of key-value pairs.
Each pair is written as ``key: value``, where whitespace after the ``:`` is optional.

This guide introduces the main syntax of YAML relevant when writing
a configuration file to use with ``SkyPy``. Essentially, it would start with
definitions of individual variables at the top, followed by the tables to produce,
and, within the table entries, the features of the objects to simulate are included.


Variables
^^^^^^^^^
* `Define a variable`: a variable is defined as a key-value pair at the top of the file. You can define any type of variable:

  .. code:: yaml

      counter: 100 # An integer
      miles: 1000.0 # A floating point
      name: "Joy" # A string
      mylist: [ 'abc', 789, 2.0e3 ] # A list


* `Reference a variable`: variables can be referenced by their full name tagged with a dollar sign ``$``.
  For example, if you previously defined a list of galaxy properties:

  .. code:: yaml

      galaxy_properties: [ 'OBJECT_ID', 'RA', 'DEC', 'REDSHIFT', 'FLUX', 'FLUX_ERR' ]

  You could reference the variable:

  .. code:: yaml

      catalog: $galaxy_properties



Functions
^^^^^^^^^
* `Call a function`: functions are defined as tuples where the first entry is the fully qualified function name tagged with and exclamation mark ``!`` and the second entry is either a list of positional arguments or a dictionary of keyword arguments.

  For example, you need to call the ``log10()`` and ``linspace()`` NumPy_ functions, for this you define the following key-value pairs:

  .. code:: yaml

      log_of_2: !numpy.log10 [2]
      myarray: !numpy.linspace [0, 2.5, 10]

  You can also define parameters of functions with a dictionary of keyword arguments.
  Imagine you want to compute the comoving distance for a range of redshifts and an `Astropy` Planck 2015 cosmology.
  To run it with the `SkyPy` pipeline, you would call the function and define the parameters as an indented dictionary

  .. code:: yaml

      comoving_distance: !astropy.cosmology.Planck15.comoving_distance
        z: !numpy.linspace [ 0, 1.3, num=10 ]

  or you could also define the variables at the top level and then reference them

  .. code:: yaml

      redshift: !numpy.linspace [ 0, 1.3, num=10 ]
      comoving_distance: !astropy.cosmology.Planck15.comoving_distance
        z: $redshift


Tables
^^^^^^

* `Create a table`: a dictionary of table names, each resolving to a dictionary of column names for that table.

  Let us create a table called ``telescope`` with a column to store the width of spectral lines that follow a normal distribution

  .. code:: yaml

      tables:
        telescope:
          spectral_lines: !scipy.stats.norm
            loc: 550
            scale: 1.6
            size: 100

* `Add a column`: you can add as many columns to a table as you need.
    Imagine you want to add a column for the telescope collecting surface

  .. code:: yaml

      tables:
        telescope:
          spectral_lines: !scipy.stats.norm
            loc: 550
            scale: 1.6
            size: 100
          collecting_surface: !numpy.random.uniform
            low: 6.9
            high: 7.1

* `Reference a column`: columns in the pipeline can be referenced by their full name tagged with a dollar sign ``$``.
  Example: the radius of cosmic voids seem to follow a lognormal distribution. You could create a table ``cosmic_voids``
  with a column ``radii`` where you sample 10000 void sizes and two other columns, ``mean`` and ``variance`` that reference
  the first column


  .. code:: yaml

    tables:
      cosmic_voids:
        radii: !scipy.stats.lognorm.rvs
          s: 200.
          size: 10000
        mean: !numpy.mean
          a: $radii
        variance: !numpy.var
          a: $radii


* `Multi-column assignment`: if a function returns multiple columns, you can chose to assign them to multiple columns with different names or to a muti-column object.

  Imagine you want the distribution of the circular velocities of 1000 halos following a Maxwellian distribution.
  The histogram NumPy_ returns a 2-dimensional object. Therefore, you could choose

  .. code:: yaml

    tables:
      halos:
        circular_velocities: !scipy.stats.maxwell.rvs
          s: 250
          size: 1000
        hist, bin_edges: !numpy.histogram
          a: $circular_velocities
          density: True

  or a multi-column assignment

  .. code:: yaml

    tables:
      halos:
        circular_velocities: !scipy.stats.maxwell.rvs
          s: 250
          size: 1000
        histogram: !numpy.histogram
          a: $circular_velocities
          density: True


* `Table.init and table.complete dependencies`:

Cosmology, a special parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Define parameters`: parameters are variables that can be modified at execution.

  For example,
  .. code:: yaml

    parameters:
      hubble_constant: 70
      omega_matter: 0.3

* The `cosmology` to be used by functions within the pipeline only needs to be set up at the top. If a function needs ``cosmology`` as an input, you need not define it again, it is automatically detected.

  .. code:: yaml

    parameters:
      hubble_constant: 70
      omega_matter: 0.3
    cosmology: !astropy.cosmology.FlatLambdaCDM
      H0: $hubble_constant
      Om0: $omega_matter



.. _YAML: https://yaml.org
.. _NumPy: https://numpy.org



Walkthrough example
-------------------

This dialog-like walkthrough example shows the natural flow of SkyPy pipelines and
how to think through the process of creating a general configuration file.
You can find more complex examples_ in our documentation.

* `SkyPy`: Hi! This is SkyPy, how can I help?
* `User`: Hi! I need to sample redshifts and magnitudes from a Schechter function. I would like to run my own pipeline within SkyPy.
* `SkyPy`: that’s brilliant! Do you have your own function or is it included in SkyPy or any other compatible package?
* `User`: I choose the SkyPy luminosity function, `~skypy.galaxies.schechter_lf`
* `SkyPy`: Nice choice! But remember you can always use other libraries, as SkyPy has the flexibility to interface with external softwares.
  The parameters for the SkyPy luminosity function, `~skypy.galaxies.schechter_lf`
  are: redshift, the characteristic absolute magnitude, the amplitude, faint-end slope parameter, the magnitude limit, the fraction of sky, cosmology and noise.
  Would you need to reuse these parameters?
* `User`: yes, all of them except for the Schechter parameters. I will also use the default value for noise.
* `SkyPy`: brill! You can define these variables at the top of your config file

  .. code:: yaml

    cosmology: !astropy.cosmology.default_cosmology.get
    z_range: !numpy.linspace [0, 2, 21]
    magnitude_limit: 23
    sky_area: 10 deg2

* `User`: I would like to create a table with a column for the blue galaxies, as I intend to also include more features later on.
* `SkyPy`: in that case, you can create the table `blue_galaxies` and for now add the columns for redshift and magnitude (note here the ``schechter_lf`` returns a 2D object)

  .. code:: yaml

    tables:
      blue_galaxies:
        redshift, magnitude: !skypy.galaxies.schechter_lf
      		redshift: $z_range
      		M_star: 20
      		phi_star: 3e-3
      		alpha: -1.3
      		m_lim: $magnitude_limit
      		sky_area: $sky_area

* `User`: Why didn’t you define the cosmology parameter?
* `SkyPy`: Aha! Good question! Remember, if cosmology is detected as parameter but is not set, it automatically uses the variable at the top of the file.
  This is how your entire config file looks like! You can now save it as ``luminosity.yml`` and run it using our SkyPy `~skypy.pipeline.Pipeline`!

  .. code:: yaml

    cosmology: !astropy.cosmology.default_cosmology.get
    z_range: !numpy.linspace [0, 2, 21]
    magnitude_limit: 23
    sky_area: 10 deg2
    tables:
      blue_galaxies:
        redshift, magnitude: !skypy.galaxies.schechter_lf
      		redshift: $z_range
      		M_star: 20
      		phi_star: 3e-3
      		alpha: -1.3
      		m_lim: $magnitude_limit
      		sky_area: $sky_area

Don’t forget to check out for more complete examples_!

.. _examples: https://skypy.readthedocs.io/en/stable/examples/index.html
