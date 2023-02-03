import numpy as np
from astropy.cosmology import Planck15
from astropy.units import allclose
from astropy import units as u
from astropy.utils.data import get_pkg_data_filename
import pytest

'''
from matplotlib import pyplot as plt
plt.ion()
plt.loglog(test_k, np.abs(test_pzk[0]/pzk[0] - 1))
'''

# load the external class result to test against
# class_result_filename = get_pkg_data_filename('data/class_result.txt')
# test_pzk = np.loadtxt(class_result_filename)

# try to import the requirement, if it doesn't exist, skip test
try:
    __import__('classy')
    CLASS_294_NOT_FOUND = __import__('classy').__version__ != 'v2.9.4'
except ImportError:
    CLASS_294_NOT_FOUND = True


@pytest.mark.skipif(CLASS_294_NOT_FOUND, reason='classy v2.9.4 not found')
def test_classy_massive():
    '''
    Test a default astropy cosmology
    '''

    truth_pk_filename = get_pkg_data_filename('data/truth_pk_massive_nu.txt')
    test_k, test_pzk0, test_pzk1 = np.loadtxt(truth_pk_filename, unpack=True)
    test_pzk = np.column_stack([test_pzk0, test_pzk1]).T

    from skypy.power_spectrum import CLASSY

    A_s, n_s, tau = 2.e-9, 0.965, 0.079
    redshift = [0.0, 1.0]
    # wavenumber = np.logspace(-4.0, np.log10(2.0), 200)
    wavenumber = test_k
    k_max = wavenumber.max()
    ps = CLASSY(k_max, redshift, Planck15, A_s, n_s, tau)
    pzk = ps(wavenumber, redshift)
    assert pzk.shape == (len(redshift), len(wavenumber))
    assert allclose(pzk, test_pzk, rtol=2.e-3)

    # also check redshifts are ordered correctly
    redshift = [1.0, 0.0]
    ps = CLASSY(k_max, redshift, Planck15, A_s, n_s, tau)
    pzk = ps(wavenumber, redshift)
    assert pzk.shape == (len(redshift), len(wavenumber))
    assert allclose(pzk, test_pzk[np.argsort(redshift)], rtol=2.e-3)

    # also check scalar arguments are treated correctly
    redshift = 1.0
    wavenumber = 1.e-1
    ps = CLASSY(k_max, redshift, Planck15, A_s, n_s, tau)
    pzk = ps(wavenumber, redshift)
    assert np.isscalar(pzk)


@pytest.mark.skipif(CLASS_294_NOT_FOUND, reason='classy v2.9.4 not found')
def test_classy_massless():
    '''
    Test a default astropy cosmology
    '''

    truth_pk_filename = get_pkg_data_filename('data/truth_pk_massless_nu.txt')
    test_k, test_pzk0, test_pzk1 = np.loadtxt(truth_pk_filename, unpack=True)
    test_pzk = np.column_stack([test_pzk0, test_pzk1]).T

    from skypy.power_spectrum import CLASSY

    Planck15massless = Planck15.clone(name='Planck 15 massless neutrino',
                                      m_nu=[0., 0., 0.]*u.eV)

    A_s, n_s, tau = 2.e-9, 0.965, 0.079
    redshift = [0.0, 1.0]
    # wavenumber = np.logspace(-4.0, np.log10(2.0), 200)
    wavenumber = test_k
    k_max = wavenumber.max()
    ps = CLASSY(k_max, redshift, Planck15massless, A_s, n_s, tau)
    pzk = ps(wavenumber, redshift)
    assert pzk.shape == (len(redshift), len(wavenumber))
    assert allclose(pzk, test_pzk, rtol=2.e-3)

    # also check redshifts are ordered correctly
    redshift = [1.0, 0.0]
    ps = CLASSY(k_max, redshift, Planck15massless, A_s, n_s, tau)
    pzk = ps(wavenumber, redshift)
    assert pzk.shape == (len(redshift), len(wavenumber))
    assert allclose(pzk, test_pzk[np.argsort(redshift)], rtol=2.e-3)

    # also check scalar arguments are treated correctly
    redshift = 1.0
    wavenumber = 1.e-1
    ps = CLASSY(k_max, redshift, Planck15massless, A_s, n_s, tau)
    pzk = ps(wavenumber, redshift)
    assert np.isscalar(pzk)
