from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
import numpy as np
import os
import pytest
from skypy.pipeline import Lightcone


def test_lightcone():

    # Test that lightcone parameters are parsed and handled correctly
    z_min, z_max, n_slice = 1, 2, 5
    nz = 100
    config = {'lightcone': {'z_min': z_min, 'z_max': z_max, 'n_slice': n_slice},
              'tables':
              {'test_table':
               {'z1': (np.random.uniform, ['$slice_z_min', '$slice_z_max', nz]),
                'z2': (np.random.uniform, ['$slice_z_mid', '$slice_z_max', nz])
                }
               }
              }

    # Sample values depending on the boundaries and midpoints of each slice
    lightcone = Lightcone(config)
    lightcone.execute()
    chi1 = lightcone.cosmology.comoving_distance(lightcone.tables['test_table']['z1'])
    chi2 = lightcone.cosmology.comoving_distance(lightcone.tables['test_table']['z2'])

    # Calculate the comoving distance to the slice boundaries and midpoints
    chi_min = lightcone.cosmology.comoving_distance(z_min)
    chi_max = lightcone.cosmology.comoving_distance(z_max)
    chi_edge = np.linspace(chi_min, chi_max, n_slice + 1)
    chi_mid = (chi_edge[:-1] + chi_edge[1:]) / 2

    # Check sampled values are consistent with slice boundaries and midpoints
    for i, (low, hi, mid) in enumerate(zip(chi_edge[:-1], chi_edge[1:], chi_mid)):
        np.testing.assert_array_less(low, chi1[i*nz:(i+1)*nz])
        np.testing.assert_array_less(chi1[i*nz:(i+1)*nz], hi)
        np.testing.assert_array_less(mid, chi2[i*nz:(i+1)*nz])
        np.testing.assert_array_less(chi2[i*nz:(i+1)*nz], hi)

    # Repeat test with non-default cosmology
    config['parameters'] = {'H0': 70, 'Om0': 0.3}
    config['cosmology'] = (FlatLambdaCDM, {'H0': '$H0', 'Om0': '$Om0'})
    lightcone = Lightcone(config)
    lightcone.execute()
    chi1 = lightcone.cosmology.comoving_distance(lightcone.tables['test_table']['z1'])
    chi2 = lightcone.cosmology.comoving_distance(lightcone.tables['test_table']['z2'])

    # Calculate the comoving distance to the slice boundaries and midpoints
    chi_min = lightcone.cosmology.comoving_distance(z_min)
    chi_max = lightcone.cosmology.comoving_distance(z_max)
    chi_edge = np.linspace(chi_min, chi_max, n_slice + 1)
    chi_mid = (chi_edge[:-1] + chi_edge[1:]) / 2

    # Check sampled values are consistent with slice boundaries and midpoints
    for i, (low, hi, mid) in enumerate(zip(chi_edge[:-1], chi_edge[1:], chi_mid)):
        np.testing.assert_array_less(low, chi1[i*nz:(i+1)*nz])
        np.testing.assert_array_less(chi1[i*nz:(i+1)*nz], hi)
        np.testing.assert_array_less(mid, chi2[i*nz:(i+1)*nz])
        np.testing.assert_array_less(chi2[i*nz:(i+1)*nz], hi)

    # Test writing to file and overwrite failure
    lightcone.write(file_format='fits')
    with fits.open('test_table.fits') as hdu:
        assert np.all(Table(hdu[1].data) == lightcone.tables['test_table'])
    with pytest.raises(OSError):
        lightcone.write(file_format='fits', overwrite=False)


def teardown_module(module):

    # Remove fits file generated in test_lightcone
    os.remove('test_table.fits')
