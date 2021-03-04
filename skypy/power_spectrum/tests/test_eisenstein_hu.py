import numpy as np
import pytest
from astropy.cosmology import Planck15, FlatLambdaCDM
from skypy.power_spectrum import eisenstein_hu


def test_eisenstein_hu():
    """ Test Eisenstein & Hu Linear matter power spectrum with
    and without wiggles using Planck15 cosmology"""
    cosmology = Planck15
    A_s = 2.1982e-09
    n_s = 0.969453
    kwmap = 0.02

    # Test that a scalar input gives a scalar output
    scalar_input = 1
    scalar_output_w = eisenstein_hu(scalar_input, A_s, n_s, cosmology, kwmap,
                                    wiggle=True)
    scalar_output_nw = eisenstein_hu(scalar_input, A_s, n_s, cosmology, kwmap,
                                     wiggle=False)
    assert np.isscalar(scalar_output_w)
    assert np.isscalar(scalar_output_nw)

    # Test that an array input gives an array output
    array_shape = (10,)
    array_input = np.random.uniform(size=array_shape)
    array_output_w = eisenstein_hu(array_input, A_s, n_s, cosmology, kwmap,
                                   wiggle=True)
    array_output_nw = eisenstein_hu(array_input, A_s, n_s, cosmology, kwmap,
                                    wiggle=False)
    assert array_output_w.shape == array_shape
    assert array_output_nw.shape == array_shape

    # Test pk against precomputed values for Planck15 cosmology
    wavenumber = np.logspace(-3, 1, num=5, base=10.0)
    pk_eisensteinhu_w = eisenstein_hu(wavenumber, A_s, n_s, cosmology, kwmap,
                                      wiggle=True)
    pk_eisensteinhu_nw = eisenstein_hu(wavenumber, A_s, n_s, cosmology, kwmap,
                                       wiggle=False)
    pk_pre_w = np.array([2.99126326e+04, 1.32023496e+05, 1.80797616e+04,
                            1.49108261e+02, 4.53912529e-01])
    pk_pre_nw = np.array([2.99050810e+04, 1.34379783e+05, 1.78224637e+04,
                            1.46439700e+02, 4.44325443e-01])

    assert np.allclose(pk_eisensteinhu_w, pk_pre_w)
    assert np.allclose(pk_eisensteinhu_nw, pk_pre_nw)

    # Test for failure when wavenumber <= 0
    negative_wavenumber_scalar = 0
    with pytest.raises(ValueError):
        eisenstein_hu(negative_wavenumber_scalar, A_s, n_s, cosmology, kwmap,
                      wiggle=True)
    with pytest.raises(ValueError):
        eisenstein_hu(negative_wavenumber_scalar, A_s, n_s, cosmology, kwmap,
                      wiggle=False)
    negative_wavenumber_array = [0, 1, -2, 3]
    with pytest.raises(ValueError):
        eisenstein_hu(negative_wavenumber_array, A_s, n_s, cosmology, kwmap,
                      wiggle=True)
    with pytest.raises(ValueError):
        eisenstein_hu(negative_wavenumber_array, A_s, n_s, cosmology, kwmap,
                      wiggle=False)

    # Test for failure when cosmology has Ob0 = 0 and wiggle = True
    zero_ob0_cosmology = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    wavenumber = np.logspace(-3, 1, num=5, base=10.0)
    with pytest.raises(ValueError):
        eisenstein_hu(wavenumber, A_s, n_s, zero_ob0_cosmology, kwmap,
                      wiggle=True)

    # Test for failure when cosmology has Tcmb = 0  and wiggle = True
    zero_Tcmb0_cosmology = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    wavenumber = np.logspace(-3, 1, num=5, base=10.0)
    with pytest.raises(ValueError):
        eisenstein_hu(wavenumber, A_s, n_s, zero_Tcmb0_cosmology, kwmap,
                      wiggle=True)
