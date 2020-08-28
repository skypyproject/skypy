"""
``skypy`` Command Line Script
=============================

``skypy`` is a command line script that runs a pipeline of functions defined in
a config file to generate tables of objects and write them to file.

Using ``skypy`` to run one of the example pipelines and write the outputs to
fits files:

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
from astropy.cosmology import z_at_value
from astropy.table import Table, vstack
from copy import deepcopy
import numpy as np
from skypy.pipeline.driver import SkyPyDriver
import sys


def main(args=None):

    import yaml

    parser = argparse.ArgumentParser(description="SkyPy pipeline driver")
    parser.add_argument('config', type=argparse.FileType('r'),
                        help='Config file name')
    parser.add_argument('-f', '--format', required=False,
                        choices=['fits', 'hdf5'], help='Table file format')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Whether to overwrite existing files')
    parser.add_argument('-l', '--lightcone', nargs=3,
                        metavar=('z_min', 'z_max', 'n_slice'),
                        help='Lightcone simulation in redshift slices')

    # get system args if none passed
    if args is None:
        args = sys.argv[1:]

    args = parser.parse_args(args or ['--help'])
    config = yaml.safe_load(args.config)
    config = {} if config is None else config

    if args.lightcone:

        # Equally spaced comoving distance slices and corresponding redshifts
        cosmology = SkyPyDriver()._call_from_config(config.get('cosmology'))
        z_min = float(args.lightcone[0])
        z_max = float(args.lightcone[1])
        n_slice = int(args.lightcone[2])
        chi_min = cosmology.comoving_distance(z_min)
        chi_max = cosmology.comoving_distance(z_max)
        chi = np.linspace(chi_min, chi_max, n_slice + 1)
        chi_mid = (chi[:-1] + chi[1:]) / 2
        z = [z_at_value(cosmology.comoving_distance, c, z_min, z_max) for c in chi[1:-1]]
        z_mid = [z_at_value(cosmology.comoving_distance, c, z_min, z_max) for c in chi_mid]
        redshift_slices = zip([z_min] + z, z + [z_max], z_mid)
        config['lightcone_z_min'] = z_min
        config['lightcone_z_max'] = z_max

        tables = {k: Table() for k in config.get('tables', {}).keys()}
        for z_min, z_max, z in redshift_slices:
            config_z = deepcopy(config)
            config_z['slice_z_min'] = z_min
            config_z['slice_z_max'] = z_max
            config_z['slice_z_mid'] = z
            driver = SkyPyDriver()
            driver.execute(config_z)
            for k, v in tables.items():
                tables[k] = vstack((v, driver[k]))
    else:
        driver = SkyPyDriver()
        driver.execute(config)
        tables = {k: driver[k] for k in config.get('tables', {}).keys()}

    # Write tables to file
    if args.format:
        for name, table in tables.items():
            filename = '.'.join((name, args.format))
            table.write(filename, overwrite=args.overwrite)

    return(0)
