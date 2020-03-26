import numpy as np
import scipy.stats

from skypy.galaxy.spectra import _spectral_coeff, sampling_coefficients


def test_spectral_coeff():
    a10 = 2.
    a11 = 3.
    z = 1.
    assert _spectral_coeff(z, a10, a11) == a11

    a10 = np.array([1., 2., 3.])
    a11 = np.array([5., 10., 14.])
    redshift = np.array([0.1, 1, 2])
    result = np.array([1.17461894, 10., 65.33333333])
    np.testing.assert_allclose(_spectral_coeff(redshift, a10, a11), result)


def test_sampling_coefficients():
    a10 = 2.079
    a20 = 3.524
    a30 = 1.917
    a40 = 1.992
    a50 = 2.536
    a11 = 2.265
    a21 = 3.862
    a31 = 1.921
    a41 = 1.685
    a51 = 2.480

    redshift = np.full(1000, 2.0, dtype=float)
    a1 = _spectral_coeff(redshift[0], a10, a11)
    a2 = _spectral_coeff(redshift[0], a20, a21)
    a3 = _spectral_coeff(redshift[0], a30, a31)
    a4 = _spectral_coeff(redshift[0], a40, a41)
    a5 = _spectral_coeff(redshift[0], a50, a51)

    a0 = a1 + a2 + a3 + a4 + a5

    coefficients = sampling_coefficients(redshift, a10, a20, a30, a40, a50,
                                         a11, a21, a31, a41, a51)
    assert coefficients.shape == (len(redshift), 5)

    # the marginalised distributions are beta distributions with a = ai and
    # b = a0-ai
    d1, p1 = scipy.stats.kstest(coefficients[:, 0], 'beta', args=(a1, a0 - a1))
    d2, p2 = scipy.stats.kstest(coefficients[:, 1], 'beta', args=(a2, a0 - a2))
    d3, p3 = scipy.stats.kstest(coefficients[:, 2], 'beta', args=(a3, a0 - a3))
    d4, p4 = scipy.stats.kstest(coefficients[:, 3], 'beta', args=(a4, a0 - a4))
    d5, p5 = scipy.stats.kstest(coefficients[:, 4], 'beta', args=(a5, a0 - a5))

    assert p1 >= 0.01 and p2 >= 0.01 and p3 >= 0.01 \
        and p4 >= 0.01 and p5 >= 0.01

    # test shape if redshift is a float
    redshift = 2.0
    coefficients = sampling_coefficients(redshift, a10, a20, a30, a40, a50,
                                         a11, a21, a31, a41, a51)
    assert coefficients.shape == (1, 5)
