from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from contextlib import redirect_stdout
from io import StringIO
import pytest
from skypy import __version__ as skypy_version
from skypy.pipeline import load_skypy_yaml
from skypy.pipeline._items import Call
from skypy.pipeline.scripts import skypy


def test_skypy(tmp_path):

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
    config_filename = get_pkg_data_filename('data/empty_config.yml')
    output_filename = str(tmp_path / 'empty.fits')
    assert skypy.main([config_filename, output_filename]) == 0

    # Process test config file
    config_filename = get_pkg_data_filename('data/test_config.yml')
    output_filename = str(tmp_path / 'test.fits')
    assert skypy.main([config_filename, output_filename]) == 0

    # Invalid file format
    output_filename = str(tmp_path / 'test.invalid')
    with pytest.raises(SystemExit) as e:
        skypy.main([config_filename, output_filename])
    assert e.value.code == 2


def test_logging(capsys, tmp_path):

    # Run skypy with default verbosity and check log is empty
    config_filename = get_pkg_data_filename('data/test_config.yml')
    output_filename = str(tmp_path / 'logging.fits')
    skypy.main([config_filename, output_filename])
    out, err = capsys.readouterr()
    assert(not err)

    # Run again with increased verbosity and capture log. Force an exception by
    # not using the "--overwrite" flag when the output file already exists.
    with pytest.raises(SystemExit):
        skypy.main([config_filename, output_filename, '--verbose'])
    out, err = capsys.readouterr()

    # Determine all DAG jobs and function calls from config
    config = load_skypy_yaml(config_filename)
    cosmology = config.pop('cosmology', None)
    tables = config.pop('tables', {})
    config.update({k: v.pop('.init', Call(Table)) for k, v in tables.items()})
    columns = [f'{t}.{c}' for t, cols in tables.items() for c in cols]
    functions = [f for f in config.values() if isinstance(f, Call)]
    functions += [f for t, cols in tables.items() for f in cols.values() if isinstance(f, Call)]

    # Check all jobs appear in the log
    for job in list(config) + list(tables) + columns:
        log_string = f"[INFO] skypy.pipeline: Generating {job}"
        assert(log_string in err)

    # Check all functions appear in the log
    for f in functions:
        log_string = f"[INFO] skypy.pipeline: Calling {f.function.__name__}"
        assert(log_string in err)

    # Check cosmology appears in the log
    if cosmology:
        assert("[INFO] skypy.pipeline: Setting cosmology" in err)

    # Check writing output file is in the log
    assert(f"[INFO] skypy: Writing {output_filename}" in err)

    # Check error for existing output file is in the log
    try:
        # New error message introduced in astropy PR #12179
        from astropy.utils.misc import NOT_OVERWRITING_MSG
        error_string = NOT_OVERWRITING_MSG.format(output_filename)
    except ImportError:
        # Fallback on old error message from astropy v4.x
        error_string = f"[ERROR] skypy: File {output_filename!r} already exists."
    assert(error_string in err)

    # Run again with decreased verbosity and check the log is empty
    with pytest.raises(SystemExit):
        skypy.main([config_filename, output_filename, '-qq'])
    out, err = capsys.readouterr()
    assert(not err)
