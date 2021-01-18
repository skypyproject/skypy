'''skypy Command Line Script'''

import argparse
from skypy import __version__ as skypy_version
from skypy.pipeline import Pipeline, load_skypy_yaml
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

    pipeline = Pipeline(config)
    pipeline.execute()
    pipeline.write(file_format=args.format, overwrite=args.overwrite)
    return(0)
