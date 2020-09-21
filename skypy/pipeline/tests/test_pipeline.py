from astropy.cosmology import default_cosmology
from astropy.io import fits
from astropy.table import Table
import networkx
import numpy as np
import os
import pytest
from skypy.pipeline import Pipeline


def setup_module(module):

    # Define function for multi-column assignment tests
    module.multi_column_array = lambda nrows, ncols: np.ones((nrows, ncols))
    module.multi_column_tuple = lambda nrows, ncols: (np.ones(nrows),) * ncols


def test_pipeline():

    # Evaluate and store the default astropy cosmology.
    config = {'test_cosmology': ('astropy.cosmology.default_cosmology.get',)}

    pipeline = Pipeline(config)
    pipeline.execute()
    assert pipeline.test_cosmology == default_cosmology.get()

    # Generate a simple two column table with a dependency. Also write the
    # table to a fits file and check it's contents.
    size = 100
    string = size*'a'
    config = {'tables': {
                'test_table': {
                  'column1': ('numpy.random.uniform', {
                      'size': size}),
                  'column2': ('numpy.random.uniform', {
                      'low': '$test_table.column1'}),
                  'column3': ('list', [string])}}}

    pipeline = Pipeline(config)
    pipeline.execute()
    pipeline.write(file_format='fits')
    assert len(pipeline.test_table) == size
    assert np.all(pipeline.test_table['column1'] < pipeline.test_table['column2'])
    with fits.open('test_table.fits') as hdu:
        assert np.all(Table(hdu[1].data) == pipeline.test_table)

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

    # Check for failure if 'column1' calls a nonexistant module
    config['tables']['test_table']['column1'] = ('nomodule.function',)
    with pytest.raises(ModuleNotFoundError):
        Pipeline(config).execute()

    # Check for failure if 'column1' calls a nonexistant function
    config['tables']['test_table']['column1'] = ('builtins.nofunction',)
    with pytest.raises(AttributeError):
        Pipeline(config).execute()

    # Check for failure if 'column1' requires itself creating a cyclic
    # dependency graph
    config['tables']['test_table']['column1'] = ('list', '$test_table.column1')
    with pytest.raises(networkx.NetworkXUnfeasible):
        Pipeline(config).execute()

    # Check for failure if 'column1' and 'column2' both require each other
    # creating a cyclic dependency graph
    config['tables']['test_table']['column1'] = ('list', '$test_table.column2')
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
    assert isinstance(pipeline.test_int, int)
    assert isinstance(pipeline.test_float, float)
    assert isinstance(pipeline.test_string, str)
    assert isinstance(pipeline.test_list, list)
    assert isinstance(pipeline.test_dict, dict)
    assert pipeline.test_int == 1
    assert pipeline.test_float == 1.0
    assert pipeline.test_string == 'hello world'
    assert pipeline.test_list == [0, 'one', 2.]
    assert pipeline.test_dict == {'a': 'b'}

    # Check variables intialised by function
    config = {'test_func': ('list', 'hello world'),
              'len_of_test_func': ('len', '$test_func'),
              'nested_references': ('sum', [
                ['$test_func', [' '], '$test_func'], []]),
              'nested_functions': ('list', ('range', ('len', '$test_func')))}
    pipeline = Pipeline(config)
    pipeline.execute()
    assert pipeline.test_func == list('hello world')
    assert pipeline.len_of_test_func == len('hello world')
    assert pipeline.nested_references == list('hello world hello world')
    assert pipeline.nested_functions == list(range(len('hello world')))


def test_multi_column_assignment():

    # Test multi-column assignment from 2d arrays and tuples of 1d arrays
    config = {'tables': {
                'multi_column_test_table': {
                  'a/b/c': ('skypy.pipeline.tests.test_pipeline.multi_column_array', [7, 3]),
                  'd/e/f': ('skypy.pipeline.tests.test_pipeline.multi_column_tuple', [7, 3])}}}

    pipeline = Pipeline(config)
    pipeline.execute()


@pytest.mark.parametrize(('na', 'nt'), [(2, 3), (4, 3), (3, 2), (3, 4)])
def test_multi_column_assignment_failure(na, nt):

    # Test multi-column assignment failure with too few/many columns
    config = {'tables': {
                'multi_column_test_table': {
                  'a/b/c': ('skypy.pipeline.tests.test_pipeline.multi_column_array', [7, na]),
                  'd/e/f': ('skypy.pipeline.tests.test_pipeline.multi_column_tuple', [7, nt])}}}

    pipeline = Pipeline(config)
    with pytest.raises(ValueError):
        pipeline.execute()


def teardown_module(module):

    # Remove fits file generated in test_pipeline
    os.remove('test_table.fits')
