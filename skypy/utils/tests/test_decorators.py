import numpy as np
import pytest


def test_uses_default_cosmology():

    from astropy.cosmology import default_cosmology, WMAP9

    from skypy.utils import uses_default_cosmology

    @uses_default_cosmology
    def function_with_cosmology(cosmology):
        return cosmology

    assert function_with_cosmology() == default_cosmology.get()

    assert WMAP9 != default_cosmology.get()
    assert function_with_cosmology(WMAP9) == WMAP9


def test_broadcast_arguments():

    from pytest import raises

    from skypy.utils import broadcast_arguments

    @broadcast_arguments('a', 'b')
    def assert_same_shape(a, b):
        assert a.shape == b.shape

    a = [1, 2, 3]
    b = [[4], [5], [6]]

    assert_same_shape(a, b)

    a = [[1, 2, 3], [7, 8, 9]]

    with raises(ValueError):
        assert_same_shape(a, b)

    with raises(ValueError):
        @broadcast_arguments('a', 'b')
        def argument_b_does_not_exist(a):
            return a


def test_dependent_argument():

    from pytest import raises

    from skypy.utils import dependent_argument

    @dependent_argument('y', 'x')
    def assert_y_is_2x(x, y):
        assert np.all(y == 2*x)

    x = np.arange(0, 1, 0.1)

    assert_y_is_2x(x, 2*x)
    assert_y_is_2x(x, lambda x: 2*x)

    @dependent_argument('z', 'x', 'y')
    def assert_z_is_2x_plus_y(x, y, z):
        assert np.all(z == 2*x+y)

    x = np.arange(0, 1, 0.1)
    y = np.arange(1, 2, 0.1)

    assert_z_is_2x_plus_y(x, y, 2*x+y)
    assert_z_is_2x_plus_y(x, y, lambda x, y: 2*x+y)

    @dependent_argument('x')
    def assert_x_is_1(x):
        assert x == 1

    assert_x_is_1(1)
    assert_x_is_1(lambda: 1)

    with raises(ValueError):
        @dependent_argument('x', 'y', 'z')
        def argument_z_does_not_exist(x, y):
            pass


def test_spectral_data_input():

    from astropy import units
    from skypy.utils import spectral_data_input

    @spectral_data_input(bandpass=units.dimensionless_unscaled)
    def my_bandpass_function(bandpass):
        pass

    my_bandpass_function('Johnson_B')

    with pytest.raises(units.UnitConversionError):
        my_bandpass_function('kcorrect_spec')

    with pytest.raises(ValueError):
        @spectral_data_input(invalid=units.Jy)
        def my_spectrum_function(spectrum):
            pass
