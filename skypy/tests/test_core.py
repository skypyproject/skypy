from astropy.cosmology import default_cosmology
from astropy.io import fits
from astropy.table import Table
import numpy as np
import os
import pytest
from skypy.core import SkyPyDriver


def test_core():

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

    # Check for failure if 'column1' is removed from the config so that the
    # dependency is broken.
    del config['tables']['test_table']['column1']
    with pytest.raises(KeyError):
        driver.execute(config)


def teardown_module(module):

    # Remove fits file generated in test_core
    os.remove('test_table.fits')
