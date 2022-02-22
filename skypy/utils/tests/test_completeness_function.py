import numpy as np


def test_logistic_completeness_function():
    from skypy.utils.completeness_function import logistic_completeness_function

    # Test that (nb, nm) array is returned if m in (mb, nm) and m95 and m50 in (nb, ) are given
    m = np.ndarray((5, 50))
    m[:] = np.linspace(18, 28, 50)
    m = m.T
    m50 = np.array([24.7, 25, 23, 23.5, 28])
    m95 = np.array([24, 23, 22, 22.3, 25])
    p = logistic_completeness_function(m, m95, m50)
    assert m.shape == p.shape

    # Test that returned shape is equal to shape of m if it is 1D array and m05 and m50 floats
    m = np.linspace(18, 28, 50)
    m50 = 24.7
    m95 = 24
    p = logistic_completeness_function(m, m95, m50)
    assert m.shape == p.shape

    # Test that returned shape is equal to shape of m if it is 2D array and m05 and m50 floats
    # array
    m = np.ndarray((5, 50))
    m[:] = np.linspace(18, 28, 50)
    m = m.T
    m50 = 24.7
    m95 = 24
    p = logistic_completeness_function(m, m95, m50)
    assert m.shape == p.shape

    # Test completeness is 0.95 for m=m95 and 0.50 for m=m50
    m95 = 24
    m50 = 25
    p = logistic_completeness_function([m95, m50], m95, m50)
    np.testing.assert_array_almost_equal(p, np.array([0.95, 0.5]))
