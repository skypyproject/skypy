from astropy.cosmology import FlatLambdaCDM, default_cosmology
from astropy.cosmology.core import Cosmology
from astropy.io import fits
from astropy.io.misc.hdf5 import read_table_hdf5
from astropy.table import Table, vstack
from astropy.table.column import Column
from astropy.units import Quantity
from astropy.utils.data import get_pkg_data_filename
import networkx
import numpy as np
import pytest
from skypy.pipeline import Pipeline
from skypy.pipeline._items import Call, Ref

try:
    import h5py # noqa
except ImportError:
    HAS_H5PY = False
else:
    HAS_H5PY = True


def test_pipeline(tmp_path):

    # Evaluate and store the default astropy cosmology.
    config = {'test_cosmology': Call(default_cosmology.get)}

    pipeline = Pipeline(config)
    pipeline.execute()
    assert pipeline['test_cosmology'] == default_cosmology.get()

    # Generate a simple two column table with a dependency. Also write the
    # table to a fits file and check it's contents.
    size = 100
    string = size*'a'
    config = {'tables': {
                'test_table': {
                  'column1': Call(np.random.uniform, [], {
                      'size': size}),
                  'column2': Call(np.random.uniform, [], {
                      'low': Ref('test_table.column1')}),
                  'column3': Call(list, [string], {})}}}

    pipeline = Pipeline(config)
    pipeline.execute()
    output_filename = str(tmp_path / 'output.fits')
    pipeline.write(output_filename)
    assert len(pipeline['test_table']) == size
    assert np.all(pipeline['test_table.column1'] < pipeline['test_table.column2'])
    with fits.open(output_filename) as hdu:
        assert np.all(Table(hdu['test_table'].data) == pipeline['test_table'])

    # Test invalid file extension
    with pytest.raises(ValueError):
        pipeline.write('output.invalid')

    # Check for failure if output files already exist and overwrite is False
    pipeline = Pipeline(config)
    pipeline.execute()
    with pytest.raises(OSError):
        pipeline.write(output_filename, overwrite=False)

    # Check that the existing output files are modified if overwrite is True
    new_size = 2 * size
    new_string = new_size*'a'
    config['tables']['test_table']['column1'].kwargs = {'size': new_size}
    config['tables']['test_table']['column3'].args = [new_string]
    pipeline = Pipeline(config)
    pipeline.execute()
    pipeline.write(output_filename, overwrite=True)
    with fits.open(output_filename) as hdu:
        assert len(hdu[1].data) == new_size

    # Check for failure if 'column1' requires itself creating a cyclic
    # dependency graph
    config['tables']['test_table']['column1'] = Call(list, [Ref('test_table.column1')])
    with pytest.raises(networkx.NetworkXUnfeasible):
        Pipeline(config).execute()

    # Check for failure if 'column1' and 'column2' both require each other
    # creating a cyclic dependency graph
    config['tables']['test_table']['column1'] = Call(list, [Ref('test_table.column2')])
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
    config = {'test_func': Call(list, ['hello world']),
              'len_of_test_func': Call(len, [Ref('test_func')]),
              'nested_references': Call(sum, [
                [Ref('test_func'), [' '], Ref('test_func')], []]),
              'nested_functions': Call(list, [Call(range, [Call(len, [Ref('test_func')])])])}
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


def test_unknown_reference():
    config = {'param1': Ref('param2')}
    pipeline = Pipeline(config)
    with pytest.raises(KeyError):
        pipeline.execute()

    config = {'mydict': {
                'param1': Ref('mydict.param2')}}
    pipeline = Pipeline(config)
    with pytest.raises(KeyError):
        pipeline.execute()

    config = {'tables': {
                'mytable': {
                  'mycolumn': [0, 1, 2]}},
              'myvalue': Ref('mytable.myothercolumn')}
    pipeline = Pipeline(config)
    with pytest.raises(KeyError):
        pipeline.execute()


def test_multi_column_assignment():

    # Test multi-column assignment from 2d arrays and tuples of 1d arrays
    config = {'tables': {
                'multi_column_test_table': {
                  'a,b ,c, d': Call(lambda nrows, ncols: np.ones((nrows, ncols)), [7, 4]),
                  'e , f,  g': Call(lambda nrows, ncols: (np.ones(nrows),) * ncols, [7, 3]),
                  'h': Call(list, [Ref('multi_column_test_table.a')]),
                  'i': Call(list, [Ref('multi_column_test_table.f')])}}}

    pipeline = Pipeline(config)
    pipeline.execute()


@pytest.mark.parametrize(('na', 'nt'), [(2, 3), (4, 3), (3, 2), (3, 4)])
def test_multi_column_assignment_failure(na, nt):

    # Test multi-column assignment failure with too few/many columns
    config = {'tables': {
                'multi_column_test_table': {
                  'a,b,c': Call(lambda nrows, ncols: np.ones((nrows, ncols)), [7, na]),
                  'd,e,f': Call(lambda nrows, ncols: (np.ones(nrows),) * ncols, [7, nt])}}}

    pipeline = Pipeline(config)
    with pytest.raises(ValueError):
        pipeline.execute()


def test_pipeline_cosmology():

    def return_cosmology(cosmology):
        return cosmology

    # Test pipeline correctly sets default cosmology from parameters
    # N.B. astropy cosmology class has not implemented __eq__ for comparison
    H0, Om0 = 70, 0.3
    config = {'parameters': {'H0': H0, 'Om0': Om0},
              'cosmology': Call(FlatLambdaCDM, [Ref('H0'), Ref('Om0')]),
              'test': Call(return_cosmology),
              }
    pipeline = Pipeline(config)
    pipeline.execute()
    assert pipeline['test'] is pipeline['cosmology']

    # Test pipeline correctly updates cosmology from new parameters
    H0_new, Om0_new = 75, 0.25
    pipeline.execute({'H0': H0_new, 'Om0': Om0_new})
    assert pipeline['test'] is pipeline['cosmology']


def test_pipeline_read():

    # Test reading config from a file
    filename = get_pkg_data_filename('data/test_config.yml')
    pipeline = Pipeline.read(filename)
    pipeline.execute()
    assert isinstance(pipeline['test_int'], int)
    assert isinstance(pipeline['test_float'], float)
    assert isinstance(pipeline['test_str'], str)
    assert isinstance(pipeline['cosmology'], Cosmology)
    assert isinstance(pipeline['test_table_1'], Table)
    assert isinstance(pipeline['test_table_1']['test_column_3'], Column)


def test_column_quantity():

    # Regression test for pull request #356
    # Previously Pipeline.__getitem__ would return column data from tables as
    # an astropy.table.Column object. However, most functions take either
    # numpy.ndarray or astropy.units.Quantity objects as arguments. As of
    # astropy version 4.1.0 Column does not support all of the same class
    # methods as Quantity e.g. to_value. This test ensures that column data in
    # a Pipeline is accessed as either an ndarray or Quantity (depending on
    # units). It also checks that functions using methods not supported by
    # Column can be called on column data inside a Pipeline.

    def value_in_cm(q):
        return q.to_value(unit='cm')

    config = {
        'tables': {
            'test_table': {
                'lengths': Quantity(np.random.uniform(size=50), unit='m'),
                'lengths_in_cm': Call(value_in_cm, [Ref('test_table.lengths')])}}}

    pipeline = Pipeline(config)
    pipeline.execute()

    assert isinstance(pipeline['test_table.lengths'], Quantity)
    assert isinstance(pipeline['test_table.lengths_in_cm'], np.ndarray)
    np.testing.assert_array_less(0, pipeline['test_table.lengths_in_cm'])
    np.testing.assert_array_less(pipeline['test_table.lengths_in_cm'], 100)


@pytest.mark.skipif(not HAS_H5PY, reason='Requires h5py')
def test_hdf5(tmp_path):
    size = 100
    string = size*'a'
    config = {'tables': {
              'test_table': {
                'column1': Call(np.random.uniform, [], {
                  'size': size}),
                'column2': Call(np.random.uniform, [], {
                  'low': Ref('test_table.column1')}),
                'column3': Call(list, [string], {})}}}

    pipeline = Pipeline(config)
    pipeline.execute()
    output_filename = str(tmp_path / 'output.hdf5')
    pipeline.write(output_filename)
    hdf_table = read_table_hdf5(output_filename, 'tables/test_table', character_as_bytes=False)
    assert np.all(hdf_table == pipeline['test_table'])


def test_depends():

    # Regression test for GitHub Issue #464
    # Previously the .depends keyword was also being passed to functions as a
    # keyword argument. This was because Pipeline was executing Item.infer to
    # handle additional function arguments from context before handling
    # additional dependencies specified using the .depends keyword. The
    # .depends keyword is now handled first.

    config = {'tables': {
                'table_1': {
                  'column1': Call(np.random.uniform, [0, 1, 10])},
                'table_2': {
                    '.init': Call(vstack, [], {
                      'tables': [Ref('table_1')],
                      '.depends': ['table_1.complete']})}}}

    pipeline = Pipeline(config)
    pipeline.execute()
    assert np.all(pipeline['table_1'] == pipeline['table_2'])
