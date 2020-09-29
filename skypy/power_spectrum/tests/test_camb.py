import numpy as np
from astropy.cosmology import Planck15
from astropy.units import allclose
from astropy.utils.data import get_pkg_data_filename
import pytest

# load the external camb result to test against
camb_result_filename = get_pkg_data_filename('data/camb_result.txt')
test_pzk = np.loadtxt(camb_result_filename)

# try to import the requirement, if it doesn't exist, skip test
try:
    __import__('camb')
except ImportError:
    CAMB_NOT_FOUND = True
else:
    CAMB_NOT_FOUND = False


@pytest.mark.skipif(CAMB_NOT_FOUND, reason='CAMB not found')
def test_camb():
    '''
    Test a default astropy cosmology
    '''
    from skypy.power_spectrum import camb

    # test shape and compare with the mocked power spectrum
    redshift = [0.0, 1.0]
    wavenumber = np.logspace(-4.0, np.log10(2.0), 200)
    pzk = camb(wavenumber, redshift, Planck15, 2.e-9, 0.965)
    assert pzk.shape == (len(redshift), len(wavenumber))
    assert allclose(pzk, test_pzk, rtol=1.e-4)

    # also check redshifts are ordered correctly
    redshift = [1.0, 0.0]
    pzk = camb(wavenumber, redshift, Planck15, 2.e-9, 0.965)
    assert pzk.shape == (len(redshift), len(wavenumber))
    assert allclose(pzk, test_pzk[np.argsort(redshift), :], rtol=1.e-4)
    