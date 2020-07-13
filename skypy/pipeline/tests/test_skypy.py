from astropy.utils.data import get_pkg_data_filename
import pytest
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

    # Missing required argument '-c'
    with pytest.raises(SystemExit) as e:
        skypy.main(['--format', 'fits'])
    assert e.value.code == 2

    # Invalid file format
    with pytest.raises(SystemExit) as e:
        skypy.main(['--config', 'config.filename', '--format', 'invalid'])
    assert e.value.code == 2

    # Process empty config file
    filename = get_pkg_data_filename('data/empty_config.yml')
    assert skypy.main(['--config', filename]) == 0

    # Process test config file
    filename = get_pkg_data_filename('data/test_config.yml')
    assert skypy.main(['--config', filename]) == 0
