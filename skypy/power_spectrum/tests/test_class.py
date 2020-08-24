import numpy as np
from astropy.cosmology import Planck15
from astropy.units import allclose
from astropy.utils.data import get_pkg_data_filename
from unittest.mock import patch, MagicMock
import pytest

# load the external class result to test against
class_result_filename = get_pkg_data_filename('data/class_result.txt')
test_pkz = np.loadtxt(class_result_filename, delimiter=',')

values = {'0': test_pkz[:,0], '1': test_pkz[:,0]}
def side_effect(arg1, arg2):
    return values[str(arg)]

# create a mock object and specify values for all the attributes needed in
# _class.py
class_mock = MagicMock()
class_mock.Class().set.return_value = 1
class_mock.Class().compute.return_value = 1
#class_mock.Class().pk = test_pkz_list.pop
class_mock.Class().pk = MagicMock(side_effect=side_effect)

# try to import the requirement, if it doesn't exist, use the mock instead
try:
    __import__('classyc')
    class_import_loc = {}
except ImportError:
    class_import_loc = {'classyc': class_mock}


@patch.dict('sys.modules', class_import_loc)
def test_classy():
    '''
    Test a default astropy cosmology
    '''
    from skypy.power_spectrum import classy

    redshift = [0.0, 1.0]
    wavenumber = np.logspace(-4.0, np.log10(2.0), 200)
    pkz = classy(wavenumber, redshift, Planck15, 2.e-9, 0.965)
    assert pkz.shape == (len(wavenumber), len(redshift))
    assert allclose(pkz, test_pkz, rtol=1.e-4)

    # also check redshifts are ordered correctly
    redshift = [1.0, 0.0]
    pkz = classy(wavenumber, redshift, Planck15, 2.e-9, 0.965)
    assert pkz.shape == (len(wavenumber), len(redshift))
    assert allclose(pkz, test_pkz[:, np.argsort(redshift)], rtol=1.e-4)
