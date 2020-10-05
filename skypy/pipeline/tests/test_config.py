from astropy.utils.data import get_pkg_data_filename
from collections.abc import Callable
import pytest
from skypy.pipeline import load_skypy_yaml


def test_load_skypy_yaml():

    # Read empty config file
    filename = get_pkg_data_filename('data/empty_config.yml')
    assert load_skypy_yaml(filename) == {}

    # Read config file and check entries are parsed to the correct types
    filename = get_pkg_data_filename('data/test_config.yml')
    config = load_skypy_yaml(filename)
    assert isinstance(config['test_int'], int)
    assert isinstance(config['test_float'], float)
    assert isinstance(config['test_str'], str)
    assert isinstance(config['test_func'], tuple)
    assert isinstance(config['test_cosmology'][0], Callable)
    assert isinstance(config['test_cosmology'][1], dict)
    assert isinstance(config['tables']['test_table_1']['test_column_3'][0], Callable)
    assert isinstance(config['tables']['test_table_1']['test_column_3'][1], list)

    # Bad function
    filename = get_pkg_data_filename('data/bad_function.yml')
    with pytest.raises(ImportError):
        load_skypy_yaml(filename)

    # Bad module
    filename = get_pkg_data_filename('data/bad_module.yml')
    with pytest.raises(ImportError):
        load_skypy_yaml(filename)
