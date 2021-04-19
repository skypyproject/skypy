###################
Configuration files
###################

This page outlines how to construct configuration files to run your own routines
with `~skypy.pipeline.Pipeline`.

`SkyPy` is an astrophysical simulation pipeline tool that allows to define any
arbitrary workflow and store data in table format. You may use `SkyPy` `~skypy.pipeline.Pipeline`.
to call any function --your own, from any compatible external software or from the `SkyPy library`.
Then `SkyPy` deals with the data dependencies and provides a library of functions to be used with it.

These guidelines start with an example using one of the `SkyPy` functions, and it follows
the concrete YAML syntax necessary for you to write your own configuration files, beyond using `SkyPy`
functions.

SkyPy example
-------------

In this section, we exemplify how you can write a configuration file and use some of the `SkyPy` functions.
In this example, we sample redshifts and magnitudes from the SkyPy luminosity function, `~skypy.galaxies.schechter_lf`.

- `Define variables`:

From the documentation, the parameters for the `~skypy.galaxies.schechter_lf` function are: ``redshift``, the characteristic absolute magnitude ``M_star``, the amplitude ``phi_star``, faint-end slope parameter ``alpha``,
the magnitude limit ``magnitude_limit``, the fraction of sky ``sky_area``, ``cosmology`` and ``noise``.
If you are planning to reuse some of these parameters, you can define them at the top-level of your configuration file.
In our example, we are using ``Astropy`` linear and exponential models for the characteristic absolute magnitude and the amplitude, respectively.
Also, ``noise`` is an optional parameter and you could use its default value by omitting its definition.

  .. code:: yaml

    cosmology: !astropy.cosmology.default_cosmology.get
    z_range: !numpy.linspace [0, 2, 21]
    M_star: !astropy.modeling.models.Linear1D [-0.9, -20.4]
    phi_star: !astropy.modeling.models.Exponential1D [3e-3, -9.7]
    magnitude_limit: 23
    sky_area: 10 deg2

- `Tables and columns`:

You can create a table ``blue_galaxies`` and for now add the columns for redshift and magnitude (note here the ``schechter_lf`` returns a 2D object)

  .. code:: yaml

    tables:
      blue_galaxies:
        redshift, magnitude: !skypy.galaxies.schechter_lf
      		redshift: $z_range
      		M_star: $M_star
      		phi_star: $phi_star
      		alpha: -1.3
      		m_lim: $magnitude_limit
      		sky_area: $sky_area

`Important:` if cosmology is detected as a parameter but is not set, it automatically uses the cosmology variable defined at the top-level of the file.

This is how the entire configuration file looks like!
You may now save it as ``luminosity.yml`` and run it using the `SkyPy` `~skypy.pipeline.Pipeline`:

.. literalinclude:: luminosity.yml
   :language: yaml

You may now save it as ``luminosity.yml`` and run it using the `SkyPy` `~skypy.pipeline.Pipeline`:

.. plot::
   :include-source: true
   :context: close-figs

    import matplotlib.pyplot as plt
    from skypy.pipeline import Pipeline

    # Execute SkyPy luminosity pipeline
    pipeline = Pipeline.read("luminosity.yml")
    pipeline.execute()

    # Blue population
    skypy_galaxies = pipeline['blue_galaxies']

    # Plot histograms
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))

    axs[0].hist(skypy_galaxies['redshift'], bins=50, histtype='step', color='purple')
    axs[0].set_xlabel(r'$Redshift$')
    axs[0].set_ylabel(r'$\mathrm{N}$')
    axs[0].set_yscale('log')

    axs[1].hist(skypy_galaxies['magnitude'], bins=50, histtype='step', color='green')
    axs[1].set_xlabel(r'$Magnitude$')
    axs[1].set_yscale('log')

    plt.show()

Don’t forget to check out for more complete examples_!

.. _examples: https://skypy.readthedocs.io/en/stable/examples/index.html


YAML syntax
-----------
YAML_ is a file format designed to be readable by both computers and humans.
Fundamentally, a file written in YAML consists of a set of key-value pairs.
Each pair is written as ``key: value``, where whitespace after the ``:`` is optional.

This guide introduces the main syntax of YAML relevant when writing
a configuration file to use with ``SkyPy``. Essentially, it begins with
definitions of individual variables at the top level, followed by the tables,
and, within the table entries, the features of objects to simulate are included.
Main keywords: ``parameters``, ``cosmology``, ``tables``.


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

  For example, if you need to call the ``log10()`` and ``linspace()`` NumPy_ functions, then you define the following key-value pairs:

  .. code:: yaml

      log_of_2: !numpy.log10 [2]
      myarray: !numpy.linspace [0, 2.5, 10]

  You can also define parameters of functions with a dictionary of keyword arguments.
  Imagine you want to compute the comoving distance for a range of redshifts and an `Astropy` Planck 2015 cosmology.
  To run it with the `SkyPy` pipeline, call the function and define the parameters as an indented dictionary.

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
  Example: the radius of cosmic voids seem to follow a lognormal distribution. You can create a table ``cosmic_voids``
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
  The histogram NumPy_ returns a 2-dimensional object. Therefore, you can choose

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


* `Referencing tables: table.init and table.complete dependencies`.

  Most of the time you would be referencing simple
  variables. However there are times when your function depends on tables. In these
  cases, you need to ensure the referenced table has the necessary content and is not empty.

  Imagine you want to perform a very simple abundance matching, i.e. painting galaxies within your halos.
  For this you can create two tables ``halos`` and ``galaxies`` storing the halo mass and galaxy luminosities.
  The idea is to stack these two tables and store it in a third table called ``matching``. For example:

  .. code:: yaml

    tables:
      halos:
        halo_mass: !mylibrary.my_halo_mass_function
          m_min: 1.0e8
          m_max: 1.0e14
      galaxies:
        luminosity: !mylibrary.my_schechter_function
          alpha: 1.20
      matching_wrong:
        match: !numpy.vstack
          tup: [ $halos, $galaxies ]

  This would probably not do what you intend.
  For example, if you have a table called ``tableA`` with columns ``c1`` and ``c2``.
  In configuration language, ``tableA`` is the name of the job.
  That means, when executing the config file, the first thing that happens is call ``tableA``, second,  call ``tableA.c1`` and third, call ``tableA.c2``.
  In our example, when you call the function ``numpy.vstack()`` and reference the tables ``$halos`` and ``$galaxies``, this is actually
  referencing the job that initialises the empty table ``halos`` and ``galaxies``.
  The potential risk is that the function could be called before the jobs ``halos`` and ``galaxies`` are finished, so the tables would be empty

  To overcome this issue you can initialise your ``matching`` table with ``init``, specify their dependences with the keyword ``depends``
  and ensure the tables are completed before calling the function with ``.complete``. The previous configuration file reads now:

  .. code:: yaml

    tables:
      halos:
        halo_mass: !mylibrary.my_halo_mass_function
          m_min: 1.0e8
          m_max: 1.0e14
      galaxies:
        luminosity: !mylibrary.my_schechter_function
          alpha: 1.20
      matching:
        init: !numpy.vstack
          tup: [ $halos, $galaxies ]
          depends: [ tup.complete ]



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
