from astropy.cosmology import FlatLambdaCDM, default_cosmology
from astropy.cosmology.core import Cosmology
from astropy.io import fits
from astropy.table import Table
from astropy.table.column import Column
from astropy.utils.data import get_pkg_data_filename
import networkx
import numpy as np
import os
import pytest
from skypy.pipeline import Pipeline


def test_pipeline():

    # Evaluate and store the default astropy cosmology.
    config = {'test_cosmology': (default_cosmology.get,)}

    pipeline = Pipeline(config)
    pipeline.execute()
    assert pipeline['test_cosmology'] == default_cosmology.get()

    # Generate a simple two column table with a dependency. Also write the
    # table to a fits file and check it's contents.
    size = 100
    string = size*'a'
    config = {'tables': {
                'test_table': {
                  'column1': (np.random.uniform, {
                      'size': size}),
                  'column2': (np.random.uniform, {
                      'low': '$test_table.column1'}),
                  'column3': (list, [string])}}}

    pipeline = Pipeline(config)
    pipeline.execute()
    pipeline.write(file_format='fits')
    assert len(pipeline['test_table']) == size
    assert np.all(pipeline['test_table.column1'] < pipeline['test_table.column2'])
    with fits.open('test_table.fits') as hdu:
        assert np.all(Table(hdu[1].data) == pipeline['test_table'])

    # Check for failure if output files already exist and overwrite is False
    pipeline = Pipeline(config)
    pipeline.execute()
    with pytest.raises(OSError):
        pipeline.write(file_format='fits', overwrite=False)

    # Check that the existing output files are modified if overwrite is True
    new_size = 2 * size
    new_string = new_size*'a'
    config['tables']['test_table']['column1'][1]['size'] = new_size
    config['tables']['test_table']['column3'][1][0] = new_string
    pipeline = Pipeline(config)
    pipeline.execute()
    pipeline.write(file_format='fits', overwrite=True)
    with fits.open('test_table.fits') as hdu:
        assert len(hdu[1].data) == new_size

    # Check for failure if 'column1' requires itself creating a cyclic
    # dependency graph
    config['tables']['test_table']['column1'] = (list, '$test_table.column1')
    with pytest.raises(networkx.NetworkXUnfeasible):
        Pipeline(config).execute()

    # Check for failure if 'column1' and 'column2' both require each other
    # creating a cyclic dependency graph
    config['tables']['test_table']['column1'] = (list, '$test_table.column2')
    with pytest.raises(networkx.NetworkXUnfeasible):
        Pipeline(config).execute()

    # Check for failure if 'column1' is removed from the config so that the
    # requirements for 'column2' are not satisfied.
    del config['tables']['test_table']['column1']
    with pytest.raises(KeyError):
        Pipeline(config).execute()

    # Check variables intialised by value
    config = {'test_int': 1,
              'test_float': 1.0,
              'test_string': 'hello world',
              'test_list': [0, 'one', 2.],
              'test_dict': {'a': 'b'}}
    pipeline = Pipeline(config)
    pipeline.execute()
    assert isinstance(pipeline['test_int'], int)
    assert isinstance(pipeline['test_float'], float)
    assert isinstance(pipeline['test_string'], str)
    assert isinstance(pipeline['test_list'], list)
    assert isinstance(pipeline['test_dict'], dict)
    assert pipeline['test_int'] == 1
    assert pipeline['test_float'] == 1.0
    assert pipeline['test_string'] == 'hello world'
    assert pipeline['test_list'] == [0, 'one', 2.]
    assert pipeline['test_dict'] == {'a': 'b'}

    # Check variables intialised by function
    config = {'test_func': (list, 'hello world'),
              'len_of_test_func': (len, '$test_func'),
              'nested_references': (sum, [
                ['$test_func', [' '], '$test_func'], []]),
              'nested_functions': (list, (range, (len, '$test_func')))}
    pipeline = Pipeline(config)
    pipeline.execute()
    assert pipeline['test_func'] == list('hello world')
    assert pipeline['len_of_test_func'] == len('hello world')
    assert pipeline['nested_references'] == list('hello world hello world')
    assert pipeline['nested_functions'] == list(range(len('hello world')))

    # Check parameter initialisation
    config = {'parameters': {
                'param1': 1.0}}
    pipeline = Pipeline(config)
    pipeline.execute()
    assert pipeline['param1'] == 1.0

    # Update parameter and re-run
    new_parameters = {'param1': 5.0}
    pipeline.execute(parameters=new_parameters)
    assert pipeline['param1'] == new_parameters['param1']


def test_multi_column_assignment():

    # Test multi-column assignment from 2d arrays and tuples of 1d arrays
    config = {'tables': {
                'multi_column_test_table': {
                  'a,b ,c, d': (lambda nrows, ncols: np.ones((nrows, ncols)), [7, 4]),
                  'e , f,  g': (lambda nrows, ncols: (np.ones(nrows),) * ncols, [7, 3]),
                  'h': (list, '$multi_column_test_table.a'),
                  'i': (list, '$multi_column_test_table.f')}}}

    pipeline = Pipeline(config)
    pipeline.execute()


@pytest.mark.parametrize(('na', 'nt'), [(2, 3), (4, 3), (3, 2), (3, 4)])
def test_multi_column_assignment_failure(na, nt):

    # Test multi-column assignment failure with too few/many columns
    config = {'tables': {
                'multi_column_test_table': {
                  'a,b,c': (lambda nrows, ncols: np.ones((nrows, ncols)), [7, na]),
                  'd,e,f': (lambda nrows, ncols: (np.ones(nrows),) * ncols, [7, nt])}}}

    pipeline = Pipeline(config)
    with pytest.raises(ValueError):
        pipeline.execute()

def test_pipeline_cosmology():

    # Define function for testing pipeline cosmology
    from skypy.utils import uses_default_cosmology
    @uses_default_cosmology
    def return_cosmology(cosmology):
        return cosmology

    # Initial default_cosmology
    initial_default = default_cosmology.get()

    # Test pipeline correctly sets default cosmology from parameters
    # N.B. astropy cosmology class has not implemented __eq__ for comparison
    H0, Om0 = 70, 0.3
    config = {'parameters': {'H0': H0, 'Om0': Om0},
              'cosmology': (FlatLambdaCDM, ['$H0', '$Om0']),
              'test': (return_cosmology, ),
              }
    pipeline = Pipeline(config)
    pipeline.execute()
    assert type(pipeline['test']) == FlatLambdaCDM
    assert pipeline['test'].H0.value == H0
    assert pipeline['test'].Om0 == Om0

    # Test pipeline correctly updates cosmology from new parameters
    H0_new, Om0_new = 75, 0.25
    pipeline.execute({'H0': H0_new, 'Om0': Om0_new})
    assert type(pipeline['test']) == FlatLambdaCDM
    assert pipeline['test'].H0.value == H0_new
    assert pipeline['test'].Om0 == Om0_new

    # Check that the astropy default cosmology is unchanged
    assert default_cosmology.get() == initial_default


def test_pipeline_read():

    # Test reading config from a file
    filename = get_pkg_data_filename('data/test_config.yml')
    pipeline = Pipeline.read(filename)
    pipeline.execute()
    assert isinstance(pipeline['test_int'], int)
    assert isinstance(pipeline['test_float'], float)
    assert isinstance(pipeline['test_str'], str)
    assert isinstance(pipeline['test_cosmology'], Cosmology)
    assert isinstance(pipeline['test_table_1'], Table)
    assert isinstance(pipeline['test_table_1']['test_column_3'], Column)


def teardown_module(module):

    # Remove fits file generated in test_pipeline
    os.remove('test_table.fits')
