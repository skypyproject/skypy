import numpy as np
from astropy.cosmology import Planck15
from astropy.units import allclose
from astropy import units
from astropy.utils.data import get_pkg_data_filename
import pytest

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
    from skypy.power_spectrum import CAMB

    # load the external camb result to test against
    truth_pk_filename = get_pkg_data_filename('data/truth_pk_massive_nu.txt')
    test_k, test_pzk0, test_pzk1 = np.loadtxt(truth_pk_filename, unpack=True)
    test_pzk = np.column_stack([test_pzk0, test_pzk1]).T

    # Setup CAMB interpolator
    k_max, z_grid = test_k.max(), np.array([0, 1])
    A_s, n_s, tau = 2.e-9, 0.965, 0.079
    ps = CAMB(k_max, z_grid, Planck15, A_s, n_s, tau)

    # test shape and compare with the mocked power spectrum
    redshift = [0.0, 1.0]
    # wavenumber = np.logspace(-4.0, np.log10(2.0), 200)
    wavenumber = test_k
    pzk = ps(wavenumber, redshift)
    assert pzk.shape == (len(redshift), len(wavenumber))
    assert allclose(pzk, test_pzk, rtol=1.e-4)

    # also check redshifts are ordered correctly
    redshift = [1.0, 0.0]
    pzk = ps(wavenumber, redshift)
    assert pzk.shape == (len(redshift), len(wavenumber))
    assert allclose(pzk, test_pzk[np.argsort(redshift), :], rtol=1.e-4)

@pytest.mark.skipif(CAMB_NOT_FOUND, reason='CAMB not found')
def test_camb_massless():
    '''
    Test a default astropy cosmology
    '''
    from skypy.power_spectrum import CAMB

    # load the external camb result to test against
    truth_pk_filename = get_pkg_data_filename('data/truth_pk_massless_nu.txt')
    test_k, test_pzk0, test_pzk1 = np.loadtxt(truth_pk_filename, unpack=True)
    test_pzk = np.column_stack([test_pzk0, test_pzk1]).T

    Planck15massless = Planck15.clone(name='Planck 15 massless neutrino',
                                      m_nu=[0., 0., 0.]*units.eV)

    # Setup CAMB interpolator
    k_max, z_grid = test_k.max(), np.array([0, 1])
    A_s, n_s, tau = 2.e-9, 0.965, 0.079
    ps = CAMB(k_max, z_grid, Planck15massless, A_s, n_s, tau)

    # test shape and compare with the mocked power spectrum
    redshift = [0.0, 1.0]
    # wavenumber = np.logspace(-4.0, np.log10(2.0), 200)
    wavenumber = test_k
    pzk = ps(wavenumber, redshift)
    assert pzk.shape == (len(redshift), len(wavenumber))
    assert allclose(pzk, test_pzk, rtol=1.e-4)

    # also check redshifts are ordered correctly
    redshift = [1.0, 0.0]
    pzk = ps(wavenumber, redshift)
    assert pzk.shape == (len(redshift), len(wavenumber))
    assert allclose(pzk, test_pzk[np.argsort(redshift), :], rtol=1.e-4)


@pytest.mark.skipif(CAMB_NOT_FOUND, reason='CAMB not found')
def test_camb_redshift_zero():
    """
    Regression test for #438
    Test that camb runs succesfully with redshift=0.0
    """
    from skypy.power_spectrum import CAMB

    # load the external camb result to test against
    truth_pk_filename = get_pkg_data_filename('data/truth_pk_massive_nu.txt')
    test_k, test_pzk0, test_pzk1 = np.loadtxt(truth_pk_filename, unpack=True)
    test_pzk = np.column_stack([test_pzk0, test_pzk1]).T

    # Setup CAMB interpolator
    k_max, z_grid = test_k.max(), np.array([0, 1])
    A_s, n_s, tau = 2.e-9, 0.965, 0.079
    ps = CAMB(k_max, z_grid, Planck15, A_s, n_s, tau)

    redshift = 0.0
    wavenumber = test_k
    pzk = ps(wavenumber, redshift)
    assert allclose(pzk, test_pzk[0], rtol=1.e-4)
