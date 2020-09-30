import numpy as np
import pytest
from scipy.stats import kstest


def test_schechter_lf():

    from pytest import raises
    from skypy.galaxy import schechter_lf

    # redshift and magnitude distributions are tested separately
    # only test that output is consistent here

    # parameters for the sampling
    z = np.linspace(0., 1., 100)
    M_star = -20
    phi_star = 1e-3
    alpha = -0.5
    m_lim = 30.
    fsky = 1/41253

    # sample redshifts and magnitudes
    z_gal, M_gal = schechter_lf(z, M_star, phi_star, alpha, m_lim, fsky)

    # check length
    assert len(z_gal) == len(M_gal)

    # turn M_star, phi_star, alpha into arrays
    z, M_star, phi_star, alpha = np.broadcast_arrays(z, M_star, phi_star, alpha)

    # sample s.t. arrays need to be interpolated
    # alpha array not yet supported
    with raises(NotImplementedError):
        z_gal, M_gal = schechter_lf(z, M_star, phi_star, alpha, m_lim, fsky)
