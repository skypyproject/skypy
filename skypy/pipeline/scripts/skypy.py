import argparse
from skypy.pipeline.driver import SkyPyDriver
import sys
import yaml

parser = argparse.ArgumentParser(description="SkyPy pipeline driver")
parser.add_argument('-c', '--config', required=True,
                    type=argparse.FileType('r'), help='Config file name')
parser.add_argument('-f', '--format', required=False,
                    choices=['fits', 'hdf5'], help='Table file format')


def main(args=None):

    args = parser.parse_args(args or sys.argv[1:] or ['--help'])
    config = yaml.safe_load(args.config)
    driver = SkyPyDriver()
    driver.execute(config, file_format=args.format)
