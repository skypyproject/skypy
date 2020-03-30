import numpy as np
import scipy.stats
import pytest

from skypy.galaxy.spectrum import dirichlet_coefficients


def test_sampling_coefficients():
    alpha0 = np.array([2.079, 3.524, 1.917, 1.992, 2.536])
    alpha1 = np.array([2.265, 3.862, 1.921, 1.685, 2.480])

    z1 = 1.
    redshift = np.full(1000, 2.0, dtype=float)
    redshift_reshape = np.atleast_1d(redshift)[:, np.newaxis]
    alpha = np.power(alpha0, 1. - redshift_reshape / z1) * \
        np.power(alpha1, redshift_reshape / z1)

    a0 = alpha.sum(axis=1)

    # Check the output shape if redshift is an array
    coefficients = dirichlet_coefficients(redshift, alpha0, alpha1, z1)
    assert coefficients.shape == (len(redshift), 5), \
        'Shape of coefficients array is not (len(redshift), 5) '

    # the marginalised distributions are beta distributions with a = alpha_i
    # and b = a0-alpha_i
    d1, p1 = scipy.stats.kstest(coefficients[:, 0], 'beta',
                                args=(alpha[:, 0], a0 - alpha[:, 0]))
    d2, p2 = scipy.stats.kstest(coefficients[:, 1], 'beta',
                                args=(alpha[:, 1], a0 - alpha[:, 1]))
    d3, p3 = scipy.stats.kstest(coefficients[:, 2], 'beta',
                                args=(alpha[:, 2], a0 - alpha[:, 2]))
    d4, p4 = scipy.stats.kstest(coefficients[:, 3], 'beta',
                                args=(alpha[:, 3], a0 - alpha[:, 3]))
    d5, p5 = scipy.stats.kstest(coefficients[:, 4], 'beta',
                                args=(alpha[:, 4], a0 - alpha[:, 4]))

    assert p1 >= 0.01 and p2 >= 0.01 and p3 >= 0.01 \
        and p4 >= 0.01 and p5 >= 0.01, \
        'Not all marginal distributions follow a beta distribution.'

    # Test output shape if redshift is a scalar
    redshift = 2.0
    coefficients = dirichlet_coefficients(redshift, alpha0, alpha1)
    assert coefficients.shape == (5,), \
        'Shape of coefficients array is not (5,) if redshift array is float.'

    # Test raising ValueError of alpha1 and alpha0 have different size
    alpha0 = np.array([1, 2, 3])
    alpha1 = np.array([4, 5])
    redshift = np.linspace(0, 2, 10)
    with pytest.raises(ValueError):
        dirichlet_coefficients(redshift, alpha0, alpha1)

    # Test that ValueError is risen if alpha0 or alpha1 is a scalar.
    alpha0 = 1.
    alpha1 = 2.
    redshift = np.linspace(0, 2, 10)

    with pytest.raises(ValueError,
                       match="alpha0 and alpha1 have to be array_like."):
        dirichlet_coefficients(redshift, alpha0, alpha1)

