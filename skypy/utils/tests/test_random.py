import numpy as np
import pytest
from scipy.stats import kstest


@pytest.mark.flaky
def test_schechter():

    from skypy.utils.random import schechter
    from skypy.utils.special import gammaincc

    def schechter_cdf_gen(alpha, x_min, x_max):
        a = gammaincc(alpha + 1, x_min)
        b = gammaincc(alpha + 1, x_max)
        return lambda x: (a - gammaincc(alpha + 1, x)) / (a - b)

    # Test the schechter function, sampling dimensionless x values
    alpha = -1.3
    x_min = 1e-10
    x_max = 1e2

    x = schechter(alpha, x_min, x_max, size=1000)

    assert np.all(x >= x_min)
    assert np.all(x <= x_max)

    # Test the distribution of galaxy properties follows the right distribution
    cdf = schechter_cdf_gen(alpha, x_min, x_max)
    _, p = kstest(x, cdf)
    assert p > 0.01

    # Test output shape when scale is a scalar
    scale = 5
    samples = schechter(alpha, x_min, x_max, scale=scale)
    assert np.shape(samples) == np.shape(scale)

    # Test output shape when scale is an array
    scale = np.random.uniform(size=10)
    samples = schechter(alpha, x_min, x_max, scale=scale)
    assert np.shape(samples) == np.shape(scale)


@pytest.mark.flaky
def test_schechter_gamma():

    from skypy.utils.random import schechter

    # when alpha > 0, x_min ≈ 0, x_max ≈ ∞, distribution is gamma
    # n.b. if alpha < 0 the distribution becomes too steep to resolve accurately
    alpha = np.random.uniform(0, 2)
    x_min = 1e-20
    x_max = 1e+20
    scale = 2.5

    x = schechter(alpha, x_min, x_max, resolution=100000, size=1000, scale=scale)

    _, p = kstest(x, 'gamma', args=(alpha+1, 0, scale))
    assert p > 0.01


def test_triaxial_axis_ratio():

    from skypy.utils.random import triaxial_axis_ratio

    # sample a single axis ratio
    q = triaxial_axis_ratio(0.8, 0.4)
    assert np.isscalar(q)

    # sample many axis ratios
    q = triaxial_axis_ratio(0.8, 0.4, size=1000)
    assert np.shape(q) == (1000,)

    # sample with explicit shape
    q = triaxial_axis_ratio(0.8, 0.4, size=(10, 10))
    assert np.shape(q) == (10, 10)

    # sample with implicit size
    q1 = triaxial_axis_ratio([0.8, 0.9], 0.4)
    q2 = triaxial_axis_ratio(0.8, [0.4, 0.5])
    assert np.shape(q1) == np.shape(q2) == (2,)

    # sample with broadcasting rule
    q = triaxial_axis_ratio([[0.6, 0.7], [0.8, 0.9]], [0.4, 0.5])
    assert np.shape(q) == (2, 2)

    # sample with random parameters and check that projection is
    # between largest and smallest possible value
    zeta, xi = np.sort(np.random.uniform(0, 1, size=(2, 1000)), axis=0)
    qmin = np.min([zeta, xi, xi/zeta], axis=0)
    qmax = np.max([zeta, xi, xi/zeta], axis=0)
    q = triaxial_axis_ratio(zeta, xi)
    assert np.all((qmax >= q) & (q >= qmin))
