from astropy.utils.data import get_pkg_data_filename
from contextlib import redirect_stdout
from io import StringIO
import os
import pytest
from skypy import __version__ as skypy_version
from skypy.pipeline import load_skypy_yaml
from skypy.pipeline.scripts import skypy


def test_skypy():

    # No arguments
    with pytest.raises(SystemExit) as e:
        skypy.main([])
    assert e.value.code == 0

    # Argparse help
    with pytest.raises(SystemExit) as e:
        skypy.main(['--help'])
    assert e.value.code == 0

    # Argparse version
    version = StringIO()
    with pytest.raises(SystemExit) as e:
        with redirect_stdout(version):
            skypy.main(['--version'])
    assert version.getvalue().strip() == skypy_version
    assert e.value.code == 0

    # Missing positional argument 'output'
    with pytest.raises(SystemExit) as e:
        skypy.main(['config.filename'])
    assert e.value.code == 2

    # Process empty config file
    filename = get_pkg_data_filename('data/empty_config.yml')
    assert skypy.main([filename, 'empty.fits']) == 0

    # Process test config file
    filename = get_pkg_data_filename('data/test_config.yml')
    assert skypy.main([filename, 'test.fits']) == 0

    # Invalid file format
    with pytest.raises(SystemExit) as e:
        skypy.main([filename, 'test.invalid'])
    assert e.value.code == 2


def test_logging(capsys):

    # Run skypy with default verbosity and check log is empty
    filename = get_pkg_data_filename('data/test_config.yml')
    output_file = 'logging.fits'
    skypy.main([filename, output_file])
    out, err = capsys.readouterr()
    assert(not err)

    # Run again with increased verbosity and capture log. Force an exception by
    # not using the "--overwrite" flag when the output file already exists.
    with pytest.raises(SystemExit):
        skypy.main([filename, output_file, '--verbose'])
    out, err = capsys.readouterr()

    # Determine all DAG jobs from config
    config = load_skypy_yaml(filename)
    tables = config.pop('tables', {})
    columns = [f'{t}.{c}' for t, cols in tables.items() for c in cols if c != '.init']

    # Check all jobs appear in the log
    for job in list(config) + list(tables) + columns:
        log_string = f"[INFO] skypy.pipeline: {job}"
        assert(log_string in err)

    # Check error for existing output file is in the log
    assert(f"[ERROR] skypy: File '{output_file}' already exists." in err)

    # Run again with decreased verbosity and check the log is empty
    with pytest.raises(SystemExit):
        skypy.main([filename, output_file, '-qq'])
    out, err = capsys.readouterr()
    assert(not err)


def teardown_module(module):

    # Remove fits file generated in test_skypy
    os.remove('empty.fits')
    os.remove('logging.fits')
    os.remove('test.fits')
