import numpy as np


def test_schecter_vdf():
    from skypy.galaxies.velocity_dispersion import schecter_vdf

    # test schecter velocity dispersion function
    vd_min = 100
    vd_max = 200
    size = 100
    samples = schecter_vdf(vd_min, vd_max, size=size)

    # test output is within limits and size
    assert np.all(samples <= vd_max)
    assert np.all(samples >= vd_min)
    assert np.sum((vd_min <= samples) & (samples <= vd_max)) == size

    # test output shape when scale is scalar
    scale = 5
    samples = schecter_vdf(vd_min, vd_max, scale=scale)
    assert np.shape(samples) == np.shape(scale)

    # Test output shape when scale is an array
    scale = np.random.uniform(size=10)
    samples = schecter_vdf(vd_min, vd_max, scale=scale)
    assert np.shape(samples) == np.shape(scale)
