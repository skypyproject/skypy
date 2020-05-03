import numpy as np

from skypy.observation import (no_filter, bandpass_filter, interpolated_filter)


def test_no_filter():
    # the filter always returns one, except for nan, so make sure it handles
    # input shapes correctly
    filt = no_filter()
    R = filt(1.0)
    assert np.isscalar(R) and R == 1.0

    R = filt([1., 2., 3.])
    assert np.shape(R) == (3,) and np.all(R == 1.0)

    R = filt([1., 2., np.nan])
    assert np.shape(R) == (3,) and np.all(R == [1., 1., 0.])


def test_bandpass_filter():
    # bandpass filters can take single or multiple bands
    filt = bandpass_filter(0, 1)
    R = filt(0.0)
    assert np.isscalar(R) and R == 1.0
    R = filt(0.5)
    assert np.isscalar(R) and R == 1.0
    R = filt(1.0)
    assert np.isscalar(R) and R == 0.0

    filt = bandpass_filter([0, 1, 2, 3], [1, 2, 3, 4])
    R = filt(0.5)
    assert np.shape(R) == (4,) and np.all(R == [1., 0., 0., 0.])
    R = filt([0.5, 1.5, 2.5, 3.5])
    assert np.shape(R) == (4, 4) and np.all(R == np.eye(4))


def test_interpolated_filter():
    # takes one or more transmission curves and interpolates
    x = [0., 1., 2., 3., 4.]
    R1 = [1., 1., 0., 0., 0.]
    R2 = [0., 1., 1., 0., 0.]
    R3 = [0., 0., 1., 1., 0.]

    filt = interpolated_filter(x, [R1, R2, R3])

    R = filt(0.5)
    assert np.shape(R) == (3,) and np.all(R == [1., 0.5, 0.])

    R = filt([0.5, 1.5, 2.5, 3.5, 4.5])
    assert np.shape(R) == (3, 5) and np.all(R == [[1.0, 0.5, 0.0, 0.0, 0.0],
                                                  [0.5, 1.0, 0.5, 0.0, 0.0],
                                                  [0.0, 0.5, 1.0, 0.5, 0.0]])
