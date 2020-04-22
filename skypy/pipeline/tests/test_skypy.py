import pytest
from skypy.pipeline.scripts import skypy


def test_skypy():

    # No arguments
    with pytest.raises(SystemExit) as e:
        skypy.main()
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
