"""
``skypy`` Command Line Script
=============================

``skypy`` is a command line script that runs a pipeline of functions defined in
a config file to generate tables of objects and write them to file.

Using ``skypy`` to run one of the example pipelines and write the outputs to
fits files::

    $ skypy --config examples/herbel_galaxies.yaml --format fits

Config Files
------------

Config files are written in yaml format. The top level should contain the
fields ``cosmology`` and/or ``tables``. ``cosmology`` should contain a
dictionary configuring a function that returns an
``astropy.cosmology.Cosmology`` object. ``tables`` should contain a set of
nested dictionaries, first giving the names of each table, then the names of
each column within each table. Each column should contain a dictionary
configuring a function that returns an array-like object.

Each step in the pipeline is configured by a dictionary specifying:

- 'function' : the name of the function
- 'module' : the name of the the module to import 'function' from
- 'args' : a list of positional arguments (by value)
- 'kwargs' : a dictionary of keyword arguments
- 'requires' : a dictionary of keyword arguments

Note that 'kwargs' specifices keyword arguments by value, wheras 'requires'
specifices the names of previous steps in the pipeline and uses their return
values as keyword arguments.
"""

import argparse
from skypy.pipeline.driver import SkyPyDriver
import sys
import yaml


def main(args=None):

    parser = argparse.ArgumentParser(description="SkyPy pipeline driver")
    parser.add_argument('-c', '--config', required=True,
                        type=argparse.FileType('r'), help='Config file name')
    parser.add_argument('-f', '--format', required=False,
                        choices=['fits', 'hdf5'], help='Table file format')

    args = parser.parse_args(args or sys.argv[1:] or ['--help'])
    config = yaml.safe_load(args.config)
    driver = SkyPyDriver()
    driver.execute(config, file_format=args.format)
