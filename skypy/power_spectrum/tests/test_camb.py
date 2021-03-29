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
    from skypy.power_spectrum import CAMB

    # Setup CAMB interpolator
    k_max, z_grid = 10, np.array([0, 1])
    A_s, n_s = 2.e-9, 0.965
    ps = CAMB(k_max, z_grid, Planck15, A_s, n_s)

    # test shape and compare with the mocked power spectrum
    redshift = [0.0, 1.0]
    wavenumber = np.logspace(-4.0, np.log10(2.0), 200)
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

    # Setup CAMB interpolator
    k_max, z_grid = 10, np.array([0, 1])
    A_s, n_s = 2.e-9, 0.965
    ps = CAMB(k_max, z_grid, Planck15, A_s, n_s)

    redshift = 0.0
    wavenumber = np.logspace(-4.0, np.log10(2.0), 200)
    pzk = ps(wavenumber, redshift)
    assert allclose(pzk, test_pzk[0], rtol=1.e-4)


@pytest.mark.skipif(CAMB_NOT_FOUND, reason='CAMB not found')
def test_camb_interpolation():
    from camb import get_matter_power_interpolator
    from inspect import signature
    from skypy.power_spectrum import camb

    # Number of redshifts above which CAMB will do interpolation
    nz_step = signature(get_matter_power_interpolator).parameters['nz_step'].default

    # Two redshift arrays. For the first CAMB will calculate the exact power
    # spectrum. The second contains all of the redshifts from the first as a
    # subset, but CAMB will calculate the power spectrum using interpolation.
    z_exact = np.linspace(0, 1, nz_step)
    z_interp = np.linspace(0, 1, 2*nz_step - 1)

    # Exact power spectrum
    wavenumber = np.logspace(-3.0, 1, 100)
    A_s, n_s = 2.2E-9, 0.97
    p_exact = camb(wavenumber, z_exact, Planck15, A_s, n_s)

    # Interpolated power spectrum at the same redshifts
    p_interp = camb(wavenumber, z_interp, Planck15, A_s, n_s)[::2]
    assert allclose(p_exact, p_interp)
