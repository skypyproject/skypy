import functools

import numpy as np
import skypy.galaxy.redshifts_herbel as redshift
from unittest.mock import patch
import pytest
import scipy.stats


def test_rescale_luminosity_limit():
    assert redshift._rescale_luminosity_limit(1, 0.4, 1.7, 7) == 48.42816796432556
    result = np.array([9.976311574844399, 10.350706743955211, 10.838520524098474, 11.884201433124382])
    result_func = redshift._rescale_luminosity_limit(np.array([0.1, 0.5, 1, 2]), 0.1, 0.74, 5)
    np.testing.assert_allclose(result_func, result)


def test_convert_abs_mag_to_lum():
    assert round(redshift._convert_abs_mag_to_lum(-22), 1) == 630957344.5


def test_cdf_redshift():
    cdf = redshift._cdf_redshift(np.array([0.01, 0.5, 1, 2]),
                                 -1.3, -0.10268436,
                                 -0.9408582, 0.00370253,
                                 -20.40492365, 10**(-0.4*-16.0))
    result = np.array([-0.0, 0.19898569, 0.43251797, 1.0])
    np.testing.assert_allclose(cdf, result)


# Test that a_m, b_m, a_phi, b_phi have the right values if a galaxy type is given
@patch('skypy.galaxy.redshifts_herbel._cdf_redshift')
def test_herbel_redshift_gal_type(_cdf_redshift):
    _cdf_redshift.side_effect = [np.linspace(0, 1, 10000)]
    redshift.herbel_redshift(3, gal_type='blue')
    # mock.assert_called_once_with() does not work cause first argument is a numpy array
    desired_parameters = [np.linspace(0.01, 2.0, 10000),
                          -1.3,
                          -0.10268436,
                          -0.9408582,
                          0.00370253,
                          -20.40492365,
                          2511886.4315095823]
    called_args = _cdf_redshift.call_args.__getitem__(0)
    np.testing.assert_allclose(called_args[0], desired_parameters[0])
    for i in range(1, len(called_args)):
        assert called_args[i] == desired_parameters[i]


# Test that a_m, b_m, a_phi, b_phi have the right values if a galaxy type is None
@patch('skypy.galaxy.redshifts_herbel._cdf_redshift')
def test_herbel_redshift_gal_type_none(_cdf_redshift):
    _cdf_redshift.side_effect = [np.linspace(0, 1, 10000)]
    redshift.herbel_redshift(3, a_m=1, b_m=2, a_phi=3, b_phi=4, alpha=5)
    desired_parameters = [np.linspace(0.01, 2.0, 10000),
                          5,
                          3,
                          1,
                          4,
                          2,
                          2511886.4315095823
                          ]
    called_args = _cdf_redshift.call_args.__getitem__(0)
    np.testing.assert_allclose(called_args[0], desired_parameters[0])
    for i in range(1, len(called_args)):
        assert called_args[i] == desired_parameters[i]


# Test that Error is returned if gal_type is None and not all Herbel parameters given
def test_herbel_redshift_exception():
    with pytest.raises(ValueError) as e:
        redshift.herbel_redshift(3, a_m=1, b_b=2)
    assert str(e.value) == 'Not all required parameters are given. '\
                           'You have to give a_m, b_m, a_phi, b_phi and alpha'


# Test whether principle of the interpolation works. Let CDF return the CDF of a Gaussian
# and sample from this. Then compare the first three moment of the returned sample with the Gaussian one
@patch('skypy.galaxy.redshifts_herbel._cdf_redshift')
def test_herbel_redshift_gauss(_cdf_redshift):
    x = np.linspace(-5, 5, 10000)
    _cdf_redshift.side_effect = [scipy.stats.norm.cdf(x)]
    sample = redshift.herbel_redshift(1000000, -5., 5.0, gal_type='blue')
    p_value = scipy.stats.kstest(sample, 'norm')[1]
    assert p_value >= 0.05


# Test that the sampling follows the Schechter function as the pdf.
def test_herbel_redshift_sampling():
    sample = redshift.herbel_redshift(1000000, gal_type='blue')
    func = functools.partial(redshift._cdf_redshift,
                             alpha=-1.3,
                             a_phi=-0.10268436,
                             a_m=-0.9408582,
                             b_phi=0.00370253,
                             b_m=-20.40492365,
                             luminosity_min=2511886.4315095823)
    p_value = scipy.stats.kstest(sample, func)[1]
    assert p_value >= 0.05
