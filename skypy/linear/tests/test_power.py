from astropy.cosmology import default_cosmology
import numpy as np
from skypy.linear.power import eisenstein_hu


def test_eisenstein_hu():
    """ Test Eisenstein & Hu Linear matter power spectrum with
    and without wiggles """
    wavenumber = np.logspace(-5, 4, num=300, base=10.0)
    cosmology = default_cosmology.get()
    A_s = 2.1982e-09
    n_s = 0.969453
    power_w = eisenstein_hu(wavenumber, A_s, n_s, cosmology, wiggle=True)
    power_n = eisenstein_hu(wavenumber, A_s, n_s, cosmology, wiggle=False)
    assert wavenumber.shape == power_w.shape
    assert wavenumber.shape == power_n.shape
