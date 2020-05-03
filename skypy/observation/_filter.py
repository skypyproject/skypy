import numpy as np
import scipy.interpolate


__all__ = [
    'no_filter',
    'bandpass_filter',
    'interpolated_filter'
]


def no_filter():
    '''No filter.

    The response of the `no_filter` function is identically unity.

    Parameters
    ----------

    Returns
    -------
    A filter function that returns `1.0` in the same shape as the input array
    for all inputs except `nan`.

    '''

    return lambda x: 1 - np.isnan(x)


def bandpass_filter(a, b):
    '''A bandpass filter.

    The bandpass filter function returns `1.0` for wavelengths `a <= lam < b`
    and `0.0` otherwise. Lowpass and highpass filters can have either bound at
    infinity.

    Multiple filter bands can be specified by passing arrays as upper and lower
    bounds. In this case, the filter function will return a transmission curve
    in the shape `(nlam, nband)`.

    Parameters
    ----------
    a : float or array_like
        Lower bound in desired units. Can be `-inf` for lowpass filters.
    b : float or array_like
        Upper bound in desired units. Can be `inf` for highpass filters.

    Returns
    -------
    A filter function that creates a bandpass filter for a single or multiple
    bands.

    '''
    def shape_expand_left(x, y):
        return (*np.ones_like(np.shape(x), int), *np.shape(y))

    def shape_expand_right(x, y):
        return (*np.shape(x), *np.ones_like(np.shape(y), int))

    return lambda x: (
            np.less_equal(np.reshape(a, shape_expand_right(a, x)),
                          np.reshape(x, shape_expand_left(a, x)))
            & np.less(np.reshape(x, shape_expand_left(b, x)),
                      np.reshape(b, shape_expand_right(b, x)))
        ).astype(float)


def interpolated_filter(x, Rx):
    '''Filter from tabulated transmission curve.

    Takes an array `x` and one or more transmission curves `Rx` and returns
    an interpolated filter function for the bands.

    Parameters
    ----------
    x : array_like of shape (nx,)
        The spectral axis for the filter, typically frequency or wavelength.
    Rx : array_like of shape (nx,) or (nfilt, nx)
        The filter response curves, either for a single or multiple bands.

    Returns
    -------
    A interpolation filter function.
    '''

    return scipy.interpolate.interp1d(x, Rx,
                                      bounds_error=False, fill_value=0.0)
