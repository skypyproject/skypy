import numpy as np
from scipy import stats
from scipy.stats.tests.common_tests import (
    check_normalization, check_moment, check_mean_expect, check_var_expect,
    check_skew_expect, check_kurt_expect, check_edge_support,
    check_random_state_property, check_pickling)

from skypy.galaxy.redshift import smail, herbel_redshift, herbel_pdf
from unittest.mock import patch
import scipy.stats
import scipy.integrate


def test_smail():
    # freeze a distribution with parameters
    args = (1.3, 2.0, 1.5)
    dist = smail(*args)

    # check that PDF is normalised
    check_normalization(smail, args, 'smail')

    # check CDF and SF
    assert np.isclose(dist.cdf(3.) + dist.sf(3.), 1.)

    # check inverse CDF and SF
    assert np.isclose(dist.ppf(dist.cdf(4.)), 4.)
    assert np.isclose(dist.isf(dist.sf(5.)), 5.)

    # check median matches parameter
    zm = np.random.rand(10)
    assert np.allclose(smail.median(zm, 2.0, 1.5), zm)

    # check moments
    m, v, s, k = dist.stats(moments='mvsk')
    check_mean_expect(smail, args, m, 'smail')
    check_var_expect(smail, args, m, v, 'smail')
    check_skew_expect(smail, args, m, v, s, 'smail')
    check_kurt_expect(smail, args, m, v, k, 'smail')
    check_moment(smail, args, m, v, 'smail')

    # check other properties
    check_edge_support(smail, args)
    check_random_state_property(smail, args)
    check_pickling(smail, args)

    # sample a single redshift
    rvs = dist.rvs()
    assert np.isscalar(rvs)

    # sample 10 reshifts
    rvs = dist.rvs(size=10)
    assert rvs.shape == (10,)

    # sample with implicit sizes
    zm, a, b = np.ones(5), np.ones((7, 5)), np.ones((13, 7, 5))
    rvs = smail.rvs(z_median=zm, alpha=a, beta=b)
    assert rvs.shape == np.broadcast(zm, a, b).shape

    # check sampling against own CDF
    D, p = stats.kstest(smail.rvs, smail.cdf, args=args, N=1000)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # check sampling, for alpha=0, beta=1, the distribution is exponential
    D, p = stats.kstest(smail.rvs(0.69315, 1e-100, 1., size=1000), 'expon')
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # check sampling, for beta=1, the distribution matches a gamma distribution
    D, p = stats.kstest(smail.rvs(2.674, 2, 1, size=1000), 'gamma', args=(3,))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # check sampling, the distribution is a generalised gamma distribution
    D, p = stats.kstest(smail.rvs(0.832555, 1, 2, size=1000),
                        'gengamma', args=(1, 2))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)


def test_herbel_pdf():
    from astropy.cosmology import FlatLambdaCDM
    cosmology = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    pdf = herbel_pdf(np.array([0.01, 0.5, 1, 2]),
                     -0.5, -0.70596888,
                     0.0035097, -0.70798041,
                     -20.37196157, cosmology, np.power(10, -0.4 * -16.0))
    result = np.array(
        [4.09063927e+04, 4.45083420e+07, 7.26629445e+07, 5.40766813e+07])
    np.testing.assert_allclose(pdf, result)


# Test whether principle of the interpolation works. Let PDF return the PDF
# of a Gaussian and sample from the corresponding CDF. Then compare the
# first three moment of the returned sample with the Gaussian one.
@patch('skypy.galaxy.redshift.herbel_pdf')
def test_herbel_redshift_gauss(herbel_pdf):
    from astropy.cosmology import FlatLambdaCDM
    cosmology = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    resolution = 100
    x = np.linspace(-5, 5, resolution)
    herbel_pdf.side_effect = [scipy.stats.norm.pdf(x)]
    sample = herbel_redshift(alpha=-1.3, a_phi=-0.10268436,
                             a_m=-0.9408582, b_phi=0.00370253,
                             b_m=-20.40492365, cosmology=cosmology,
                             low=-5., high=5.0, size=1000000,
                             resolution=resolution)
    p_value = scipy.stats.kstest(sample, 'norm')[1]
    assert p_value >= 0.01


# Test that the sampling follows the pdf of Schechter function.
def test_herbel_redshift_sampling():
    from astropy.cosmology import FlatLambdaCDM
    cosmology = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    sample = herbel_redshift(alpha=-1.3, a_phi=-0.10268436,
                             a_m=-0.9408582, b_phi=0.00370253,
                             b_m=-20.40492365, cosmology=cosmology,
                             size=1000)

    def calc_cdf(z):
        pdf = herbel_pdf(z, alpha=-1.3,
                         a_phi=-0.10268436,
                         a_m=-0.9408582,
                         b_phi=0.00370253,
                         b_m=-20.40492365,
                         cosmology=cosmology,
                         luminosity_min=2511886.4315095823)
        cdf = scipy.integrate.cumtrapz(pdf, z, initial=0)
        cdf = cdf / cdf[-1]
        return cdf

    p_value = scipy.stats.kstest(sample, calc_cdf)[1]
    assert p_value >= 0.01
