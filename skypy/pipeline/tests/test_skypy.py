from astropy.utils.data import get_pkg_data_filename
from contextlib import redirect_stdout
from io import StringIO
import pytest
from skypy import __version__ as skypy_version
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

    # Missing positional argument 'config'
    with pytest.raises(SystemExit) as e:
        skypy.main(['--format', 'fits'])
    assert e.value.code == 2

    # Invalid file format
    with pytest.raises(SystemExit) as e:
        skypy.main(['--format', 'invalid', 'config.filename'])
    assert e.value.code == 2

    # Process empty config file
    filename = get_pkg_data_filename('data/empty_config.yml')
    assert skypy.main([filename]) == 0

    # Process test config file
    filename = get_pkg_data_filename('data/test_config.yml')
    assert skypy.main([filename]) == 0

    # Process lightcone config file
    filename = get_pkg_data_filename('data/lightcone_config.yml')
    assert skypy.main([filename]) == 0
