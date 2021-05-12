'''skypy Command Line Script'''

import argparse
import logging
from skypy import __version__ as skypy_version
from skypy.pipeline import Pipeline, load_skypy_yaml
import sys


def main(args=None):

    parser = argparse.ArgumentParser(description="SkyPy pipeline driver")
    parser.add_argument('--version', action='version', version=skypy_version)
    parser.add_argument('config', help='Config file name')
    parser.add_argument('output', help='Output file name')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Whether to overwrite existing files')
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase logging verbosity")
    parser.add_argument("-q", "--quiet", action="count", default=0,
                        help="Decrease logging verbosity")

    # get system args if none passed
    if args is None:
        args = sys.argv[1:]

    args = parser.parse_args(args or ['--help'])

    # Setup skypy logger
    default_level = logging._nameToLevel['WARNING']
    logging_level = default_level + 10 * (args.quiet - args.verbose)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger('skypy')
    logger.setLevel(logging_level)
    logger.addHandler(stream_handler)

    try:
        config = load_skypy_yaml(args.config)
        pipeline = Pipeline(config)
        pipeline.execute()
        if args.output:
            logger.info(f"Writing {args.output}")
            pipeline.write(args.output, overwrite=args.overwrite)
    except Exception as e:
        logger.exception(e)
        raise SystemExit(2) from e

    return(0)
