import numpy as np


def test_logistic_completeness_function():
    from skypy.utils.completeness_function import logistic_completeness_function

    # Test that (nm, nb) array is returned if magnitude in (nm, nb) and
    # magnitude_95 and magnitude_50 in (nb, ) are given
    magnitude = np.ndarray((5, 50))
    magnitude[:] = np.linspace(18, 28, 50)
    magnitude = np.reshape(magnitude, (50, 5))
    magnitude_50 = np.array([24.7, 25, 23, 23.5, 28])
    magnitude_95 = np.array([24, 23, 22, 22.3, 25])
    p = logistic_completeness_function(magnitude, magnitude_95, magnitude_50)
    assert magnitude.shape == p.shape

    # Test that returned shape is equal to shape of magnitude if it is 1D array
    # and magnitude_95 and magnitude_50 scalars
    magnitude = np.linspace(18, 28, 50)
    magnitude_50 = 24.7
    magnitude_95 = 24
    p = logistic_completeness_function(magnitude, magnitude_95, magnitude_50)
    assert magnitude.shape == p.shape

    # Test that returned shape is equal to shape of magnitude_50
    # and m_agnitude_95 if magnitude is a scalar
    magnitude = 20
    magnitude_50 = np.array([24.7, 25, 23, 23.5, 28])
    magnitude_95 = np.array([24, 23, 22, 22.3, 25])
    p = logistic_completeness_function(magnitude, magnitude_95, magnitude_50)
    assert p.shape == magnitude_50.shape == magnitude_95.shape

    # Test that returned shape is equal to shape of magnitude_50
    # if magnitude_95 and magnitude are scalar
    magnitude = 20
    magnitude_50 = np.array([24.7, 25, 23, 23.5, 28])
    magnitude_95 = 21
    p = logistic_completeness_function(magnitude, magnitude_95, magnitude_50)
    assert p.shape == magnitude_50.shape

    # Test that returned shape is equal to shape of magnitude_95
    # if magnitude_50 and magnitude are scalar
    magnitude = 20
    magnitude_50 = 28
    magnitude_95 = np.array([24, 23, 22, 22.3, 25])
    p = logistic_completeness_function(magnitude, magnitude_95, magnitude_50)
    assert p.shape == magnitude_95.shape

    # Test that returned shape is equal to shape of magnitude if it is 2D array
    # and magnitude_95 and magnitude_50 are scalar
    magnitude = np.ndarray((5, 50))
    magnitude[:] = np.linspace(18, 28, 50)
    magnitude = np.reshape(magnitude, (50, 5))
    magnitude_50 = 24.7
    magnitude_95 = 24
    p = logistic_completeness_function(magnitude, magnitude_95, magnitude_50)
    assert magnitude.shape == p.shape

    # Test completeness is 0.95 for mmagnitude=magnitude_95
    # and 0.5 for magnitude=magnitude_50
    magnitude_95 = 24
    magnitude_50 = 25
    p = logistic_completeness_function([magnitude_95, magnitude_50], magnitude_95, magnitude_50)
    np.testing.assert_array_almost_equal(p, np.array([0.95, 0.5]))
