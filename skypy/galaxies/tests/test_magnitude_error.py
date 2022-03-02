import numpy as np


def test_rykoff_error():
    from skypy.galaxies.magnitude_error import rykoff_error

    # test correct value returned
    magnitude = 30
    magnitude_limit = 25
    magnitude_zp = 30
    a = 0.5
    b = 1.0
    error = rykoff_error(magnitude, magnitude_limit, magnitude_zp, a, b)
    assert error == 10.79749285683173

    # test returned array same shape as magnitude if other parameters are scalar
    magnitude = np.linspace(15, 35, 100)
    magnitude_limit = 25
    magnitude_zp = 30
    a = 0.5
    b = 1.0
    error = rykoff_error(magnitude, magnitude_limit, magnitude_zp, a, b)
    assert error.shape == magnitude.shape

    magnitude = np.zeros((3, 100))
    magnitude[:] = np.linspace(15, 35, 100)
    magnitude_limit = 25
    magnitude_zp = 30
    a = 0.5
    b = 1.0
    error = rykoff_error(magnitude, magnitude_limit, magnitude_zp, a, b)
    assert error.shape == magnitude.shape

    # test returned array same shape as magnitude_limit if other parameters are scalar
    magnitude = 30
    magnitude_limit = np.linspace(15, 35, 100)
    magnitude_zp = 30
    a = 0.5
    b = 1.0
    error = rykoff_error(magnitude, magnitude_limit, magnitude_zp, a, b)
    assert error.shape == magnitude_limit.shape

    magnitude = 30
    magnitude_limit = np.zeros((3, 100))
    magnitude_limit[:] = np.linspace(15, 35, 100)
    magnitude_zp = 30
    a = 0.5
    b = 1.0
    error = rykoff_error(magnitude, magnitude_limit, magnitude_zp, a, b)
    assert error.shape == magnitude_limit.shape

    # test returned array same shape as magnitude_zp if other parameters are scalar
    magnitude = 30
    magnitude_limit = 25
    magnitude_zp = np.linspace(30, 35, 100)
    a = 0.5
    b = 1.0
    error = rykoff_error(magnitude, magnitude_limit, magnitude_zp, a, b)
    assert error.shape == magnitude_zp.shape

    magnitude = 30
    magnitude_limit = 25
    magnitude_zp = np.zeros((3, 100))
    magnitude_zp[:] = np.linspace(30, 35, 100)
    a = 0.5
    b = 1.0
    error = rykoff_error(magnitude, magnitude_limit, magnitude_zp, a, b)
    assert error.shape == magnitude_zp.shape

    # test returned array same shape as a if other parameters are scalar
    magnitude = 30
    magnitude_limit = 25
    magnitude_zp = 30
    a = np.linspace(0.5, 0.6, 100)
    b = 1.0
    error = rykoff_error(magnitude, magnitude_limit, magnitude_zp, a, b)
    assert error.shape == a.shape

    magnitude = 30
    magnitude_limit = 25
    magnitude_zp = 30
    a = np.zeros((3, 100))
    a[:] = np.linspace(0.5, 0.6, 100)
    b = 1.0
    error = rykoff_error(magnitude, magnitude_limit, magnitude_zp, a, b)
    assert error.shape == a.shape

    # test that returned shape is same as magnitude shape if one of the other inputs is
    # 1-d array
    magnitude_limit = np.linspace(25, 28, 20)
    magnitude = np.zeros((len(magnitude_limit), 100))
    magnitude[:] = np.linspace(15, 35, 100)
    magnitude = np.reshape(magnitude, (100, len(magnitude_limit)))
    magnitude_zp = 30
    a = 0.5
    b = 1.0
    error = rykoff_error(magnitude, magnitude_limit, magnitude_zp, a, b)
    assert error.shape == magnitude.shape

    # test that error limit is returned if error is larger than error_limit
    # compare to first test where same input returns error of 10.79749285683173
    magnitude = 30
    magnitude_limit = 25
    magnitude_zp = 30
    a = 0.5
    b = 1.0
    error_limit = 1
    error = rykoff_error(magnitude, magnitude_limit, magnitude_zp, a, b, error_limit)
    assert error == error_limit
