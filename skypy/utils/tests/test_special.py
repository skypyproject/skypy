import numpy as np
import numpy.testing as npt
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import raises

from skypy.utils.special import gammaincc


def test_gammaincc_scalar():
    # test with scalar input and expect scalar output
    npt.assert_allclose(gammaincc(1.2, 1.5), 0.28893139222051745)


def test_gammaincc_array():
    # test with vector a
    npt.assert_allclose(gammaincc([-1.2, 1.2], 1.5),
                        [0.008769092458747352, 0.28893139222051745])

    # test with vector x
    npt.assert_allclose(gammaincc(1.2, [0.5, 1.5]),
                        [0.6962998597584569, 0.28893139222051745])

    # test with vector a and x
    npt.assert_allclose(gammaincc([1.2, -1.2], [0.5, 1.5]),
                        [0.6962998597584569, 0.008769092458747352])

    # test with broadcast
    npt.assert_allclose(gammaincc([[1.2], [-1.2]], [0.5, 1.5]),
                        [[0.6962998597584569, 0.28893139222051745],
                         [0.1417443053403289, 0.008769092458747352]])


def test_gammaincc_edge_cases():
    # gammaincc is zero for x = inf
    assert gammaincc(1.2, np.inf) == 0
    assert gammaincc(-1.2, np.inf) == 0
    assert gammaincc(0, np.inf) == 0
    npt.assert_equal(gammaincc([1.2, 2.2], np.inf), [0, 0])
    npt.assert_equal(gammaincc([-1.2, -2.2], np.inf), [0, 0])
    npt.assert_equal(gammaincc([0.0, 1.0], np.inf), [0, 0])

    # gammaincc is zero for a = -1, -2, -3, ...
    assert gammaincc(-1.0, 0.5) == 0
    assert gammaincc(-2.0, 0.5) == 0
    npt.assert_equal(gammaincc([-1.0, -2.0], 0.5), [0, 0])

    # gammaincc is unity for a > 0 and x = 0
    assert gammaincc(0.5, 0) == 1
    assert gammaincc(1.5, 0) == 1
    npt.assert_equal(gammaincc([0.5, 1.5], 0), [1, 1])

    # gammaincc is zero for a nonpositive integer and x = 0
    assert gammaincc(0, 0) == 0
    assert gammaincc(-1, 0) == 0
    assert gammaincc(-2, 0) == 0
    npt.assert_equal(gammaincc([0, -1, -2], 0), [0, 0, 0])

    # gammaincc is infinity for a negative noninteger and x = 0
    assert gammaincc(-0.5, 0) == -np.inf
    assert gammaincc(-1.5, 0) == np.inf
    assert gammaincc(-2.5, 0) == -np.inf
    npt.assert_equal(gammaincc([-0.5, -1.5], 0), [-np.inf, np.inf])


def test_gammaincc_precision():
    # test precision against precomputed values
    values_file = get_pkg_data_filename('data/gammaincc.txt')
    a, x, v = np.loadtxt(values_file).T
    g = gammaincc(a, x)

    # collect where the assertion will fail before it does,
    # so we can have a more informative message
    fail = ~np.isclose(g, v, rtol=1e-10, atol=0)
    lines = []
    if np.any(fail):
        for numbers in zip(a[fail], x[fail], g[fail], v[fail]):
            lines.append('gammaincc(%g, %g) = %g != %g)' % numbers)

    # now do the assertion
    npt.assert_allclose(g, v, rtol=1e-10, atol=0, err_msg='\n'.join(lines))


@raises(ValueError)
def test_gammaincc_neg_x_scalar():
    # negative x raises an exception
    gammaincc(0.5, -1.0)


@raises(ValueError)
def test_gammaincc_neg_x_array():
    # negative x in array raises an exception
    gammaincc(0.5, [3.0, 2.0, 1.0, 0.0, -1.0])
