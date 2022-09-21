from skypy.galaxies import vdf_sampler
import numpy as np


def test_vdf_sampler():
    samples = vdf_sampler.sample_vdf(100, 200, size=1000)
    assert np.sum((100 <= samples) & (samples <= 200)) == 1000
