import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.units import allclose, eV
from unittest.mock import patch, MagicMock

from skypy.linear.tests.camb_result import camb_direct_pk_z0, camb_direct_pk_z1

# create a mock object and specify values for all the attributes needed in camb.py
camb_mock = MagicMock()
camb_mock.__version__ = 'Mock'
camb_mock.__file__ = 'test_camb.py'
camb_mock.get_results().get_matter_power_spectrum.return_value = [0, 1, np.array([camb_direct_pk_z0, camb_direct_pk_z1])]

# try to import the requirement, if it doesn't exist, use the mock instead
try:
    import camb
    camb_import_loc = {}
except ModuleNotFoundError:
    camb_import_loc = {'camb': camb_mock}


@patch.dict('sys.modules', camb_import_loc)
def test_camb():
    '''
    Test a FlatLambdaCDM cosmology with the default camb parameters
    '''
    from skypy.linear.camb import camb

    camb_default_cosmology = FlatLambdaCDM(H0=67.5,
                                           Om0=(0.122 + 0.022) / (0.675 * 0.675),
                                           Tcmb0=2.7255,
                                           Ob0=0.022 / (0.675 * 0.675),
                                           m_nu=[0.0, 0.0, 0.06] * eV)
    redshift = [0.0, 1.0]
    wavenumber = np.logspace(-4.0, np.log10(2.0), 200)  # camb wavenumbers are kh
    pk = camb(wavenumber, redshift, camb_default_cosmology, 2.e-9, 0.965)
    assert allclose(pk[0,:], camb_direct_pk_z0, rtol=1.e-4)
    assert allclose(pk[1,:], camb_direct_pk_z1, rtol=1.e-4)

    # also check redshifts are ordered correctly
    redshift = [1.0, 0.0]
    pk = camb(wavenumber, redshift, camb_default_cosmology, 2.e-9, 0.965)
    assert allclose(pk[0,:], camb_direct_pk_z1, rtol=1.e-4)
    assert allclose(pk[1,:], camb_direct_pk_z0, rtol=1.e-4)
