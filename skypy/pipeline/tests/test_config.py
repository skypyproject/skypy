from astropy.utils.data import get_pkg_data_filename
import pytest
from skypy.pipeline import load_skypy_yaml
from skypy.pipeline._items import Call
from astropy import units
from astropy.cosmology.core import Cosmology


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
    assert isinstance(config['test_func'], Call)
    assert isinstance(config['test_func_with_arg'], Call)
    assert isinstance(config['test_object'], Cosmology)
    assert isinstance(config['cosmology'], Call)
    assert isinstance(config['tables']['test_table_1']['test_column_3'], Call)

    # Bad function
    filename = get_pkg_data_filename('data/bad_function.yml')
    with pytest.raises(ImportError):
        load_skypy_yaml(filename)

    # Bad module
    filename = get_pkg_data_filename('data/bad_module.yml')
    with pytest.raises(ImportError):
        load_skypy_yaml(filename)

    # Bad object
    filename = get_pkg_data_filename('data/bad_object.yml')
    with pytest.raises(ValueError):
        load_skypy_yaml(filename)


def test_empty_ref():
    filename = get_pkg_data_filename('data/test_empty_ref.yml')
    with pytest.raises(ValueError, match='empty reference'):
        load_skypy_yaml(filename)


def test_yaml_quantities():
    # config with quantities
    filename = get_pkg_data_filename('data/quantities.yml')
    config = load_skypy_yaml(filename)

    assert config['42_km'] == units.Quantity('42 km')
    assert config['1_deg2'] == units.Quantity('1 deg2')


@pytest.mark.parametrize('config', ['numeric_key',
                                    'numeric_nested_key',
                                    'numeric_kwarg'])
def test_keys_must_be_strings(config):
    filename = get_pkg_data_filename(f'data/{config}.yml')
    with pytest.raises(ValueError, match='key ".*" is of non-string type ".*"'):
        load_skypy_yaml(filename)
