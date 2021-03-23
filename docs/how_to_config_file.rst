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
  For example, if you previously defined a list of countries:

  .. code:: yaml

      country_list: [ 'UK', 'Spain', 'Zambia', 'Chile', 'Japan' ]

  You would reference the variable;

  .. code:: yaml

      countries: $country_list



Functions
^^^^^^^^^
* `Call a function`: functions are defined as tuples where the first entry is the fully qualified function name tagged with and exclamation mark ``!`` and the second entry is either a list of positional arguments or a dictionary of keyword arguments.

  For example, you need to call the ``log10()`` and ``linspace()`` NumPy_ functions, for this you define the following key-value pairs:

  .. code:: yaml

      log_of_2: !numpy.log10 [2]
      myarray: !numpy.linspace [0, 2.5, 10]


* `Define parameters`: parameters are variables that can be modified at execution. You can also define parameters of functions with a dictionary of keyword arguments.

  Imagine you want to compute the total expense when buying a house (£230000) and a car (£15589.3). To run it with the `SkyPy` pipeline, you would call the function and define the parameters as an indented dictionary

  .. code:: yaml

      expense: !numpy.add
        x1: 230000
        x2: 15589.3

  or you could also define the variables at the top and then reference them

  .. code:: yaml

      house_price: 230000
      car_price: 15589.3
      expense: !numpy.add
        x1: $house_price
        x2: $car_price


Tables
^^^^^^
A dictionary of table names, each resolving to a dictionary of column names for that table.

* Create a table
* Add a column
* Multi-column assignment
* Table.init and table.complete dependencies

Cosmology, a special parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The cosmology to be used by functions within the pipeline.

.. _YAML: https://yaml.org
.. _NumPy: https://numpy.org



Walkthrough example
-------------------

This walkthrough example shows the natural flow of SkyPy pipelines and
how to think through the process of creating a general configuration file.
You can find more complex examples_ in our documentation.


.. _examples: https://skypy.readthedocs.io/en/stable/examples/index.html
