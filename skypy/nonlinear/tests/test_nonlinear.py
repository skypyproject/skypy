from astropy.cosmology import default_cosmology
import numpy as np
from skypy.nonlinear.power import halofit


def test_halofit():
    k = np.logspace(0.01, 10.0, 7)
    z = np.linspace(0.0, 4.0, 9)
    p = np.ones((k.size, z.size))
    c = default_cosmology.get()
    assert halofit(k, z, p, c).shape == (k.size, z.size)
