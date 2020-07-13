import numpy as np

from skypy.galaxy.redshift import smail, herbel_redshift, herbel_pdf
from unittest.mock import patch
from scipy.stats import kstest, norm
import scipy.integrate


def test_smail():
    # sample a single redshift
    rvs = smail(1.3, 2.0, 1.5)
    assert np.isscalar(rvs)

    # sample 10 reshifts
    rvs = smail(1.3, 2.0, 1.5, size=10)
    assert rvs.shape == (10,)

    # sample with implicit sizes
    zm, a, b = np.ones(5), np.ones((7, 5)), np.ones((13, 7, 5))
    rvs = smail(z_median=zm, alpha=a, beta=b)
    assert rvs.shape == np.broadcast(zm, a, b).shape

    # check sampling, for alpha=0, beta=1, the distribution is exponential
    D, p = kstest(smail(0.69315, 1e-100, 1., size=1000), 'expon')
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # check sampling, for beta=1, the distribution matches a gamma distribution
    D, p = kstest(smail(2.674, 2, 1, size=1000), 'gamma', args=(3,))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # check sampling, the distribution is a generalised gamma distribution
    D, p = kstest(smail(0.832555, 1, 2, size=1000), 'gengamma', args=(1, 2))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)


def test_herbel_pdf():
    from astropy.cosmology import FlatLambdaCDM
    cosmology = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    pdf = herbel_pdf(np.array([0.01, 0.5, 1, 2]),
                     -0.5, -0.70596888,
                     0.0035097, -0.70798041,
                     -20.37196157, cosmology, -16.0)
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
    herbel_pdf.side_effect = [norm.pdf(x)]
    sample = herbel_redshift(alpha=-1.3, a_phi=-0.10268436,
                             a_m=-0.9408582, b_phi=0.00370253,
                             b_m=-20.40492365, cosmology=cosmology,
                             low=-5., high=5.0, size=1000000,
                             resolution=resolution)
    p_value = kstest(sample, 'norm')[1]
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
                         absolute_magnitude_max=-16)
        cdf = scipy.integrate.cumtrapz(pdf, z, initial=0)
        cdf = cdf / cdf[-1]
        return cdf

    p_value = kstest(sample, calc_cdf)[1]
    assert p_value >= 0.01
