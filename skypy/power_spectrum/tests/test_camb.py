import numpy as np
from astropy.cosmology import Planck15
from astropy.units import allclose
from astropy.utils.data import get_pkg_data_filename
from unittest.mock import patch, MagicMock

# load the external camb result to test against
camb_result_filename = get_pkg_data_filename('data/camb_result.txt')
mock_pkz = np.loadtxt(camb_result_filename, delimiter=',')

# create a mock object and specify values for all the attributes needed in
# camb.py
camb_mock = MagicMock()
camb_result = [0, 1, mock_pkz.T]
camb_mock.get_results().get_matter_power_spectrum.return_value = camb_result

# try to import the requirement, if it doesn't exist, use the mock instead
try:
    __import__('camb')
    camb_import_loc = {}
except ImportError:
    camb_import_loc = {'camb': camb_mock}


@patch.dict('sys.modules', camb_import_loc)
def test_camb():
    '''
    Test a default astropy cosmology
    '''
    from skypy.power_spectrum import camb

    redshift = [0.0, 1.0]
    wavenumber = np.logspace(-4.0, np.log10(2.0), 200)
    pkz = camb(wavenumber, redshift, Planck15, 2.e-9, 0.965)
    assert pkz.shape == (len(wavenumber), len(redshift))
    assert allclose(pkz, mock_pkz, rtol=1.e-4)

    # also check redshifts are ordered correctly
    redshift = [1.0, 0.0]
    pkz = camb(wavenumber, redshift, Planck15, 2.e-9, 0.965)
    assert pkz.shape == (len(wavenumber), len(redshift))
    assert allclose(pkz, mock_pkz[:, np.argsort(redshift)], rtol=1.e-4)
