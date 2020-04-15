import numpy as np
import numpy.testing as npt

from skypy.galaxy.redshift import smail, herbel_redshift, herbel_pdf
from skypy.stats import check_rv
from unittest.mock import patch
import scipy.stats
import scipy.integrate


def test_smail():
    # check median of r.v. matches passed parameter
    zm = np.random.rand(10)
    npt.assert_allclose(smail.median(zm, 2.0, 1.5), zm)

    # standard r.v. checks
    check_rv(smail, (1.3, 2.0, 1.5), {
        # for alpha=0, beta=1, the distribution is exponential
        (0.69315, 1e-100, 1.): ['expon', ()],
        # for beta=1, the distribution matches a gamma distribution
        (2.674, 2., 1.): ['gamma', (3,)],
        # the distribution is a generalised gamma distribution
        (0.832555, 1., 2.): ['gengamma', (1., 2.)]
    })


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
