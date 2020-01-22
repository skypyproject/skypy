# ADR 1: Considering options for the SkyPy `Model`
January 22, 2020

## Context
Within SkyPy all functions used to create a "simulation" will in practice be taking in some values (either parameters or columns from a table) and creating new column(s) in an output table *or* selecting specific rows from an input table.

The inputs and outputs of these functions are clearly defined so a directed acyclic graph (DAG) can be constructed to determine what order the functions should be run in.

To aid in the creation of the tables and the DAG a helper class or decorator should be used so the person writing the function does not have to worry about the implementation details. This class or decorator is what we are currently referring to as the `Model`.

## Decision Drivers
- Ease of use: if there is too much boiler plate `Model`s will be annoying to write
- Clarity of implementation: the base `Model` should be easy to read, understand, and debug

## Considered Options

### A base `Model` class
In this implementation all functions must be written inside a class that inherits from the base `Model` class.  A different base class should be used depending on if the function adds a column to a table or selects rows.

The `__init__` method would define all the inputs and outputs and the inherited `__init__` can add this to the DAG.

The `compute` method will contain the custom function.

The `execute` method will call the `compute` method and add the results to the table/mask out rows.

- Ease of use: medium (lots of boiler plate)
- Clarity of implementation: high (Classes are well understood by most developers)

### A `Model` decorator
In this implementation all functions must use an `@Model(inputs=[], outputs=[])` decorator. A different decorator should be written for adding columns and selecting rows. The decorator will:

1. Add the `inputs` and `outputs` to the DAG
2. Return a callable function that executes the wrapped function and add the results to the table/mask out rows.

- Ease of use: easy (one line added above a function)
- Clarity of implementation: medium (decorators are functions that return function that return function... This particular implementation will be at least 3 wrappers deep)

### Use the DAG directly
Packages such as [pyungo](https://pypi.org/project/pyungo/) have APIs for most of the functionality we need here with decorators that define `inputs` and `outputs`.  When the compute graph is called all `inputs` and `outputs` are stored and returned `results` data structure.  Once computed we can write a function that turns this into the final data table.

We kind of get masking for free here as the DAG does not care if/when the number of rows changes, we just have to be careful when constructing the final table out of the `results`.

- Ease of use: easy (one line added above a function)
- Clarity of implementation: high (we off load this to an existing package that we don't have to maintain)

## Decision Outcome
Pending...