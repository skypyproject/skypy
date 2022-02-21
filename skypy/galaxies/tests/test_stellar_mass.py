import numpy as np
import scipy.stats
import scipy.integrate
from scipy.special import gammaln
import pytest

from skypy.galaxies import stellar_mass
from skypy.utils import special


@pytest.mark.flaky
def test_exponential_distribution():
    # When alpha=0, M*=1 and x_min~0 we get a truncated exponential
    q_max = 1e2
    sample = stellar_mass.schechter_smf_mass(0, 0, 1, size=1000, m_min=1e-10,
                                             m_max=q_max, resolution=1000)
    d, p_value = scipy.stats.kstest(sample, 'truncexpon', args=(q_max,))
    assert p_value >= 0.01


@pytest.mark.flaky
def test_stellar_masses():
    # Test that error is returned if m_star input is an array but size !=
    # None and size != m_star,size
    with pytest.raises(ValueError):
        stellar_mass.schechter_smf_mass(0., -1.4, np.array([1e10, 2e10]), 1.e7, 1.e13, size=3)

    # Test that an array with the sme shape as m_star is returned if m_star is
    # an array and size = None
    m_star = np.array([1e10, 2e10])
    sample = stellar_mass.schechter_smf_mass(0., -1.4, m_star, 1.e7, 1.e13, size=None)
    assert m_star.shape == sample.shape

    # Test that sampling corresponds to sampling from the right pdf.
    # For this, we sample an array of luminosities for redshift z = 1.0 and we
    # compare it to the corresponding cdf.

    def calc_pdf(m, alpha, mass_star, mass_min, mass_max):
        lg = gammaln(alpha + 1)
        c = np.fabs(special.gammaincc(alpha + 1, mass_min / mass_star))
        d = np.fabs(special.gammaincc(alpha + 1, mass_max / mass_star))
        norm = np.exp(lg) * (c - d)
        return 1. / mass_star * np.power(m / mass_star, alpha) * \
            np.exp(-m / mass_star) / norm

    def calc_cdf(m):
        alpha = -1.4
        mass_star = 10 ** 10.67
        mass_min = 10 ** 7
        mass_max = 10 ** 13
        pdf = calc_pdf(m, alpha, mass_star, mass_min, mass_max)
        cdf = scipy.integrate.cumtrapz(pdf, m, initial=0)
        cdf = cdf / cdf[-1]
        return cdf

    m_star = 10 ** 10.67
    m_min = 10 ** 7
    m_max = 10 ** 13
    sample = stellar_mass.schechter_smf_mass(0., -1.4, m_star, m_min, m_max,
                                             size=1000, resolution=100)
    p_value = scipy.stats.kstest(sample, calc_cdf)[1]
    assert p_value >= 0.01


def test_schechter_smf_parameters():
    from skypy.galaxies.stellar_mass import schechter_smf_parameters
    # Check scalar inputs
    blue = (10**-2.423, 10**10.60, -1.21)
    fsat_scalar, frho = 0.4, 0.2
    sp_scalar = schechter_smf_parameters(blue, fsat_scalar, frho)

    assert type(sp_scalar) == dict, 'active_parameters is not a dict: {}'.format(type(sp_scalar))
    assert len(sp_scalar) == 4, 'The length of the tuple is not four: {}'.format(len(sp_scalar))
    for p in sp_scalar:
        for k in range(3):
            assert type(sp_scalar[p][k]) is not np.ndarray, \
                '{} tuple is not a scalar {}'.format(p, type(sp_scalar[p][k]))

    # Check array input for the satellite fraction
    fsat_array = np.array([0.4, 0.5])
    sp_array = schechter_smf_parameters(blue, fsat_array, frho)
    assert type(sp_array) is dict, 'active_parameters is not a dict: {}'.format(type(sp_array))
    assert len(sp_array) == 4, 'The length of the tuple is not four: {}'.format(len(sp_array))
    for p in sp_array:
        assert type(sp_array[p][0]) == np.ndarray, \
            '{} amplitude is not an array {}'.format(p, type(sp_array[p][0]))
        assert len(sp_array[p][0]) == len(fsat_array), \
            '{} amplitude does not match input length {}'.format(p, len(fsat_array))
        for k in range(1, 3):
            assert type(sp_array[p][k]) != np.ndarray, \
                '{} slope or mstar not a scalar {}'.format(p, type(sp_array[p][k]))

    # Check array input for the satellite fraction
    amplitude_array = np.array([10**-2.4, 10**-2.3])
    blue_array = (amplitude_array, 10**10.60, -1.21)
    sp_array = schechter_smf_parameters(blue_array, fsat_scalar, frho)
    assert type(sp_array) is dict, 'active_parameters is not a dict: {}'.format(type(sp_array))
    assert len(sp_array) == 4, 'The length of the tuple is not four: {}'.format(len(sp_array))
    for p in sp_array:
        assert type(sp_array[p][0]) is np.ndarray, \
            '{} amplitude is not an array {}'.format(p, type(sp_array[p][0]))
        assert len(sp_array[p][0]) is len(fsat_array), \
            '{} amplitude does not match input length {}'.format(p, len(amplitude_array))
        for k in range(1, 3):
            assert type(sp_array[p][k]) is not np.ndarray, \
                '{} slope or mstar not a scalar {}'.format(p, type(sp_array[p][k]))

    # Corner cases
    # Case I: no satellite galaxies
    sp_sat0 = schechter_smf_parameters(blue, 0, frho)
    assert sp_sat0['centrals'][0] == sp_sat0['mass_quenched'][0] == blue[0], \
        'The amplitude of centrals and mass-quenched are not equal to the \
         amplitude of the blue sample'
    assert sp_sat0['satellites'][0] == sp_sat0['satellite_quenched'][0] == 0, \
        'The amplitude of satellites and satellite-quenched are not zero'

    # Case II: no satellite-quenched galaxies
    sp_rho0 = schechter_smf_parameters(blue, fsat_scalar, 0)
    assert - sp_rho0['satellite_quenched'][0] == 0, \
        'The satellite-quenched is not zero {}'.format(sp_rho0['satellite_quenched'][0])
