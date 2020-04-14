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
    config = {'cosmology': {
                'module': 'astropy.cosmology',
                'function': 'default_cosmology.get'}}

    driver = SkyPyDriver()
    driver.execute(config)
    assert driver.cosmology == default_cosmology.get()

    # Generate a simple two column table with a dependency. Also write the
    # table to a fits file and check it's contents.
    config = {'tables': {
                'test_table': {
                  'column1': {
                    'module': 'numpy.random',
                    'function': 'uniform',
                    'kwargs': {
                      'size': 100}},
                  'column2': {
                    'module': 'numpy.random',
                    'function': 'uniform',
                    'requires': {
                      'low': 'test_table.column1'}}}}}

    driver = SkyPyDriver()
    driver.execute(config, file_format='fits')
    assert len(driver.test_table) == 100
    assert np.all(driver.test_table['column1'] < driver.test_table['column2'])
    with fits.open('test_table.fits') as hdu:
        assert np.all(Table(hdu[1].data) == driver.test_table)

    # Check for failure if 'column1' requires itself creating a cyclic
    # dependency graph
    config['tables']['test_table']['column1']['requires'] = \
        {'low': 'test_table.column1'}
    with pytest.raises(networkx.NetworkXUnfeasible):
        driver.execute(config)

    # Check for failure if 'column1' and 'column2' both require each other
    # creating a cyclic dependency graph
    config['tables']['test_table']['column1']['requires'] = \
        {'low': 'test_table.column2'}
    with pytest.raises(networkx.NetworkXUnfeasible):
        driver.execute(config)

    # Check for failure if 'column1' is removed from the config so that the
    # requirements for 'column2' are not satisfied.
    del config['tables']['test_table']['column1']
    with pytest.raises(KeyError):
        driver.execute(config)


def teardown_module(module):

    # Remove fits file generated in test_driver
    os.remove('test_table.fits')
