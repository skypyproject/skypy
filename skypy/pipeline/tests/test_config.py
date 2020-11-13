from astropy.utils.data import get_pkg_data_filename
from collections.abc import Callable
import pytest
from skypy.pipeline import load_skypy_yaml
from astropy import units


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


def test_yaml_quantities():
    # config with quantities
    filename = get_pkg_data_filename('data/quantities.yml')
    config = load_skypy_yaml(filename)

    assert config['42_km'] == units.Quantity('42 km')
    assert config['1_deg2'] == units.Quantity('1 deg2')


def test_keys_must_be_strings():
    # config with keys that doesn't parse as String.
    filename = get_pkg_data_filename('data/numeric_key.yml')
    with pytest.raises(ValueError):
        config = load_skypy_yaml(filename)


def test_nested_keys_must_be_strings():
    # config with keys that doesn't parse as String.
    filename = get_pkg_data_filename('data/numeric_nested_key.yml')
    with pytest.raises(ValueError):
        config = load_skypy_yaml(filename)


def test_kwarg_must_be_strings():
    # config with kwarg that doesn't parse as String.
    filename = get_pkg_data_filename('data/numeric_kwarg.yml')
    with pytest.raises(ValueError):
        config = load_skypy_yaml(filename)
