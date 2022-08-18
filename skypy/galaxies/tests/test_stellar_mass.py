import numpy as np
import scipy.stats
import scipy.integrate
from scipy.special import gammaln
import pytest

from hypothesis import given
from hypothesis.strategies import integers
from astropy.modeling.models import Exponential1D

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

    # Test that an array with the same shape as m_star is returned if m_star is
    # an array and size = None
    m_star = np.array([1e10, 2e10])
    sample = stellar_mass.schechter_smf_mass(0., -1.4, m_star, 1.e7, 1.e13, size=None)
    assert m_star.shape == sample.shape

    # Test m_star can be a function that returns an array of values for each redshift
    redshift = np.linspace(0, 2, 100)
    alpha = 0.357
    m_min = 10 ** 7
    m_max = 10 ** 14
    m_star_function = Exponential1D(10**10.626, np.log(10)/0.095)
    sample = stellar_mass.schechter_smf_mass(redshift, alpha, m_star_function, m_min, m_max)
    assert sample.shape == redshift.shape

    # Test alpha can be a function returning a scalar value.
    sample = stellar_mass.schechter_smf_mass(redshift, lambda z: alpha, 10**10.626, m_min, m_max)
    assert sample.shape == redshift.shape

    # Sampling with an array for alpha is not implemented
    alpha_array = np.full_like(redshift, alpha)
    with pytest.raises(NotImplementedError, match='only scalar alpha is supported'):
        sample = stellar_mass.schechter_smf_mass(redshift, alpha_array, m_star, m_min, m_max)

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


def test_schechter_smf_phi_centrals():
    # Scalar inputs
    phiblue_scalar = 10**-2.423
    fsat_scalar = 0.4

    # Array inputs
    phiblue_array = np.array([10**-2.423, 10**-2.422])
    fsat_array = np.array([0.40, 0.41, 0.42])

    # Test for scalar output
    phic_scalar = stellar_mass.schechter_smf_phi_centrals(phiblue_scalar, fsat_scalar)
    assert np.isscalar(phic_scalar)

    # Test for 1 dim output
    phic_1d_phib = stellar_mass.schechter_smf_phi_centrals(phiblue_array, fsat_scalar)
    phic_1d_fsat = stellar_mass.schechter_smf_phi_centrals(phiblue_scalar, fsat_array)
    assert phic_1d_phib.shape == phiblue_array.shape
    assert phic_1d_fsat.shape == fsat_array.shape

    # Test for 2 dim output
    phic_2d = stellar_mass.schechter_smf_phi_centrals(phiblue_array[:, np.newaxis], fsat_array)
    assert phic_2d.shape == (len(phiblue_array), len(fsat_array))

    # Special case
    fsat_special = 1 - np.exp(-1)
    phic_special = stellar_mass.schechter_smf_phi_centrals(phiblue_scalar, fsat_special)
    assert phic_special == 0.5 * phiblue_scalar


@given(integers(), integers())
def test_schechter_smf_phi_mass_quenched(phic, phis):

    # Array inputs
    phic_1d = np.array([phic, phic])
    phis_1d = np.array([phis, phis])

    # Test for scalar output
    phimq_scalar = stellar_mass.schechter_smf_phi_mass_quenched(phic, phis)
    assert np.isscalar(phimq_scalar)
    assert phimq_scalar == phic + phis

    # Test for array output
    phimq_1d = stellar_mass.schechter_smf_phi_mass_quenched(phic_1d, phis_1d)
    assert phimq_1d.shape == phic_1d.shape == phis_1d.shape
    assert np.all(phimq_1d == phic_1d + phis_1d)


SATELLITE_FUNCTIONS = [
    stellar_mass.schechter_smf_phi_satellites,
    stellar_mass.schechter_smf_phi_satellite_quenched,
]


@pytest.mark.parametrize('satellite_function', SATELLITE_FUNCTIONS)
def test_schechter_smf_phi_satellites_common(satellite_function):
    # Scalar inputs
    phis_scalar = 10**-2.423
    fraction_scalar = 0.2

    # Array inputs
    phis_array = np.array([10**-2.423, 10**-2.422])

    # Test for scalar output
    phis_sat_scalar = satellite_function(phis_scalar, fraction_scalar)
    assert np.isscalar(phis_sat_scalar)

    # Test for 1 dim output
    phis_sat_1d_phib = satellite_function(phis_array, fraction_scalar)
    assert phis_sat_1d_phib.shape == phis_array.shape

    # Corner case no satellite galaxies
    fraction_null = 0
    phis_sat_null_scalar = satellite_function(phis_scalar, fraction_null)
    phis_sat_null_array = satellite_function(phis_array, fraction_null)
    assert phis_sat_null_scalar == 0
    assert np.all(phis_sat_null_array == np.zeros(len(phis_array)))
