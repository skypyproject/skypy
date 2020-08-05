from astropy.cosmology import default_cosmology
from astropy.io import fits
from astropy.table import Table
import networkx
import numpy as np
import os
import pytest
from skypy.pipeline.driver import SkyPyDriver


def test_driver():

    # Evaluate and store the default astropy cosmology.
    config = {'test_cosmology': {
                'function': 'astropy.cosmology.default_cosmology.get'}}

    driver = SkyPyDriver()
    driver.execute(config)
    assert driver.test_cosmology == default_cosmology.get()

    # Generate a simple two column table with a dependency. Also write the
    # table to a fits file and check it's contents.
    size = 100
    config = {'tables': {
                'test_table': {
                  'column1': {
                    'function': 'numpy.random.uniform',
                    'args': {
                      'size': size}},
                  'column2': {
                    'function': 'numpy.random.uniform',
                    'args': {
                      'low': 'test_table.column1'}}}}}

    driver = SkyPyDriver()
    driver.execute(config, file_format='fits')
    assert len(driver.test_table) == size
    assert np.all(driver.test_table['column1'] < driver.test_table['column2'])
    with fits.open('test_table.fits') as hdu:
        assert np.all(Table(hdu[1].data) == driver.test_table)

    # Check for failure if output files already exist and overwrite is False
    driver = SkyPyDriver()
    with pytest.raises(OSError):
        driver.execute(config, file_format='fits', overwrite=False)

    # Check that the existing output files are modified if overwrite is True
    new_size = 2 * size
    config['tables']['test_table']['column1']['args']['size'] = new_size
    driver = SkyPyDriver()
    driver.execute(config, file_format='fits', overwrite=True)
    with fits.open('test_table.fits') as hdu:
        assert len(hdu[1].data) == new_size

    # Check for failure if 'column1' requires itself creating a cyclic
    # dependency graph
    config['tables']['test_table']['column1']['args'] = {'low': 'test_table.column1'}
    with pytest.raises(networkx.NetworkXUnfeasible):
        SkyPyDriver().execute(config)

    # Check for failure if 'column1' and 'column2' both require each other
    # creating a cyclic dependency graph
    config['tables']['test_table']['column1']['args'] = {'low': 'test_table.column2'}
    with pytest.raises(networkx.NetworkXUnfeasible):
        SkyPyDriver().execute(config)

    # Check for failure if 'column1' is removed from the config so that the
    # requirements for 'column2' are not satisfied.
    del config['tables']['test_table']['column1']
    with pytest.raises(KeyError):
        SkyPyDriver().execute(config)

    # Check variables intialised by value
    config = {'test_int': 1,
              'test_float': 1.0}
    driver = SkyPyDriver()
    driver.execute(config)
    assert isinstance(driver.test_int, int)
    assert isinstance(driver.test_float, float)
    assert driver.test_int == 1
    assert driver.test_float == 1.0


def teardown_module(module):

    # Remove fits file generated in test_driver
    os.remove('test_table.fits')
