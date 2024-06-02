import numpy as np
import scipy.stats
import scipy.integrate
from scipy.special import gammaln
import pytest
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
        cdf = scipy.integrate.cumulative_trapezoid(pdf, m, initial=0)
        cdf = cdf / cdf[-1]
        return cdf

    m_star = 10 ** 10.67
    m_min = 10 ** 7
    m_max = 10 ** 13
    sample = stellar_mass.schechter_smf_mass(0., -1.4, m_star, m_min, m_max,
                                             size=1000, resolution=100)
    p_value = scipy.stats.kstest(sample, calc_cdf)[1]
    assert p_value >= 0.01
