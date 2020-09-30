import numpy as np
from astropy.cosmology import Planck15
from astropy.units import allclose
from astropy import units as u
from astropy.utils.data import get_pkg_data_filename
import pytest

# load the external class result to test against
class_result_filename = get_pkg_data_filename('data/class_result.txt')
test_pzk = np.loadtxt(class_result_filename)

# try to import the requirement, if it doesn't exist, skip test
try:
    __import__('classy')
except ImportError:
    CLASS_NOT_FOUND = True
else:
    CLASS_NOT_FOUND = False


@pytest.mark.skipif(CLASS_NOT_FOUND, reason='classy not found')
def test_classy():
    '''
    Test a default astropy cosmology
    '''
    from skypy.power_spectrum import classy

    Pl15massless = Planck15.clone(name='Planck 15 massless neutrino', m_nu=[0., 0., 0.]*u.eV)

    redshift = [0.0, 1.0]
    wavenumber = np.logspace(-4.0, np.log10(2.0), 200)
    pzk = classy(wavenumber, redshift, Pl15massless)
    assert pzk.shape == (len(redshift), len(wavenumber))
    assert allclose(pzk, test_pzk, rtol=1.e-4)

    # also check redshifts are ordered correctly
    redshift = [1.0, 0.0]
    pzk = classy(wavenumber, redshift, Pl15massless)
    assert pzk.shape == (len(redshift), len(wavenumber))
    assert allclose(pzk, test_pzk[np.argsort(redshift)], rtol=1.e-4)

    # also check scalar arguments are treated correctly
    redshift = 1.0
    wavenumber = 1.e-1
    pzk = classy(wavenumber, redshift, Pl15massless)
    assert np.isscalar(pzk)
    