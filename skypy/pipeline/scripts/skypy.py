"""
``skypy`` Command Line Script
=============================

``skypy`` is a command line script that runs a pipeline of functions defined in
a config file to generate tables of objects and write them to file.

Using ``skypy`` to run one of the example pipelines and write the outputs to
fits files:

    $ skypy examples/mccl_galaxies.yml --format fits

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
from skypy import __version__ as skypy_version
from skypy.pipeline import Lightcone, Pipeline, load_skypy_yaml
import sys


def main(args=None):

    parser = argparse.ArgumentParser(description="SkyPy pipeline driver")
    parser.add_argument('--version', action='version', version=skypy_version)
    parser.add_argument('config', help='Config file name')
    parser.add_argument('-f', '--format', required=False,
                        choices=['fits', 'hdf5'], help='Table file format')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Whether to overwrite existing files')

    # get system args if none passed
    if args is None:
        args = sys.argv[1:]

    args = parser.parse_args(args or ['--help'])
    config = load_skypy_yaml(args.config)

    if 'lightcone' in config:
        pipeline = Lightcone(config)
    else:
        pipeline = Pipeline(config)
    pipeline.execute()
    pipeline.write(file_format=args.format, overwrite=args.overwrite)
    return(0)
