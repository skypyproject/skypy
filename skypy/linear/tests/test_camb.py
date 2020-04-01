import numpy as np
from astropy.cosmology import default_cosmology
from astropy.units import allclose, eV
from unittest.mock import patch, MagicMock

from skypy.linear.tests.camb_result import camb_direct_pk_z0, camb_direct_pk_z1

# create a mock object and specify values for all the attributes needed in
# camb.py
camb_mock = MagicMock()
camb_mock.get_results().get_matter_power_spectrum.return_value = [0, 1, np.array([camb_direct_pk_z0, camb_direct_pk_z1])]

# try to import the requirement, if it doesn't exist, use the mock instead
try:
    import camb
    camb_import_loc = {}
except ImportError:
    camb_import_loc = {'camb': camb_mock}


@patch.dict('sys.modules', camb_import_loc)
def test_camb():
    '''
    Test a default astropy cosmology
    '''
    from skypy.linear.camb import camb

    redshift = [0.0, 1.0]
    wavenumber = np.logspace(-4.0, np.log10(2.0), 200)
    pk = camb(wavenumber, redshift, default_cosmology.get(), 2.e-9, 0.965)
    assert pk.shape == (len(wavenumber), len(redshift))
    assert allclose(pk[:, 0], camb_direct_pk_z0, rtol=1.e-4)
    assert allclose(pk[:, 1], camb_direct_pk_z1, rtol=1.e-4)

    # also check redshifts are ordered correctly
    redshift = [1.0, 0.0]
    pk = camb(wavenumber, redshift, default_cosmology.get(), 2.e-9, 0.965)
    assert pk.shape == (len(wavenumber), len(redshift))
    assert allclose(pk[:, 0], camb_direct_pk_z1, rtol=1.e-4)
    assert allclose(pk[:, 1], camb_direct_pk_z0, rtol=1.e-4)
