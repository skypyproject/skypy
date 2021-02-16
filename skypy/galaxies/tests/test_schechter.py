import numpy as np
from astropy.cosmology import default_cosmology


def test_schechter_lf():

    from pytest import raises
    from skypy.galaxies import schechter_lf
    from astropy import units

    # redshift and magnitude distributions are tested separately
    # only test that output is consistent here

    # parameters for the sampling
    z = np.linspace(0., 1., 100)
    M_star = -20
    phi_star = 1e-3
    alpha = -0.5
    m_lim = 30.
    sky_area = 1.0 * units.deg**2
    cosmo = default_cosmology.get()

    # sample redshifts and magnitudes
    z_gal, M_gal = schechter_lf(z, M_star, phi_star, alpha, m_lim, sky_area, cosmo)

    # check length
    assert len(z_gal) == len(M_gal)

    # turn M_star, phi_star, alpha into arrays
    z, M_star, phi_star, alpha = np.broadcast_arrays(z, M_star, phi_star, alpha)

    # sample s.t. arrays need to be interpolated
    # alpha array not yet supported
    with raises(NotImplementedError):
        z_gal, M_gal = schechter_lf(z, M_star, phi_star, alpha, m_lim, sky_area, cosmo)


def test_schechter_smf():

    from pytest import raises
    from skypy.galaxies import schechter_smf
    from astropy import units

    # redshift and magnitude distributions are tested separately
    # only test that output is consistent here

    # parameters for the sampling
    z = np.linspace(0., 1., 100)
    m_star = 10 ** 10.67
    phi_star = 1e-3
    alpha = -1.5
    m_min = 1.e7
    m_max = 1.e13
    sky_area = 1.0 * units.deg**2
    cosmo = default_cosmology.get()

    # sample redshifts and magnitudes
    z_gal, m_gal = schechter_smf(z, m_star, phi_star, alpha, m_min, m_max, sky_area, cosmo)

    # check length
    assert len(z_gal) == len(m_gal)

    # turn m_star, phi_star, alpha into arrays
    z, m_star, phi_star, alpha = np.broadcast_arrays(z, m_star, phi_star, alpha)

    # sample s.t. arrays need to be interpolated
    # alpha array not yet supported
    with raises(NotImplementedError):
        z_gal, m_gal = schechter_smf(z, m_star, phi_star, alpha, m_min, m_max, sky_area, cosmo)
