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
* Call a function
* Define parameters: Variables that can be modified at execution

Tables
^^^^^^
A dictionary of table names, each resolving to a dictionary of column names for that table

* Create a table
* Add a column
* Multi-column assignment
* Table.init and table.complete dependencies

Cosmology, a special parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The cosmology to be used by functions within the pipeline.

.. _YAML: https://yaml.org



Walkthrough example
-------------------

This walkthrough example shows the natural flow of SkyPy pipelines and
how to think through the process of creating a general configuration file.
You can find more complex examples_ in our documentation.


.. _examples: https://skypy.readthedocs.io/en/stable/examples/index.html
