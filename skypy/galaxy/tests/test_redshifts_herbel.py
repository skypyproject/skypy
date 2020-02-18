import numpy as np
import skypy.galaxy.redshifts_herbel as redshift
from unittest.mock import patch
import scipy.stats
import scipy.integrate


def test_pdf():
    pdf = redshift._pdf(np.array([0.01, 0.5, 1, 2]),
                        -0.5, -0.70596888,
                        -0.70798041, 0.0035097,
                        -20.37196157, 10 ** (-0.4 * -16.0))
    result = np.array(
        [4.09063927e+04, 4.45083420e+07, 7.26629445e+07, 5.40766813e+07])
    np.testing.assert_allclose(pdf, result)


# Test whether principle of the interpolation works. Let PDF return the PDF
# of a Gaussian and sample from the corresponding CDF. Then compare the
# first three moment of the returned sample with the Gaussian one.
@patch('skypy.galaxy.redshifts_herbel._pdf')
def test_herbel_redshift_gauss(_pdf):
    x = np.linspace(-5, 5, 10000)
    _pdf.side_effect = [scipy.stats.norm.pdf(x)]
    sample = redshift.herbel_redshift(alpha=-1.3, a_phi=-0.10268436,
                                      a_m=-0.9408582, b_phi=0.00370253,
                                      b_m=-20.40492365, low=-5.,
                                      high=5.0, size=1000000)
    p_value = scipy.stats.kstest(sample, 'norm')[1]
    assert p_value >= 0.01


# Test that the sampling follows the pdf of Schechter function.
def test_herbel_redshift_sampling():
    sample = redshift.herbel_redshift(alpha=-1.3, a_phi=-0.10268436,
                                      a_m=-0.9408582, b_phi=0.00370253,
                                      b_m=-20.40492365, size=1000)

    def calc_cdf(z):
        pdf = redshift._pdf(z, alpha=-1.3,
                            a_phi=-0.10268436,
                            a_m=-0.9408582,
                            b_phi=0.00370253,
                            b_m=-20.40492365,
                            luminosity_min=2511886.4315095823)
        cdf = scipy.integrate.cumtrapz(pdf, z, initial=0)
        cdf = cdf / cdf[-1]
        return cdf

    p_value = scipy.stats.kstest(sample, calc_cdf)[1]
    assert p_value >= 0.01
