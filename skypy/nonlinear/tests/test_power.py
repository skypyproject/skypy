
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.units import allclose
from skypy.nonlinear.power import halofit


def test_halofit():
    """ Test a FlatLambdaCDM cosmology at redshift 0 """
    # Inputs from camb linear
    k = np.array([1.00000000e-04, 1.80870267e-03, 3.27140534e-02,
                  5.91699957e-01, 1.01000000e+01])
    z = 0.0
    p = np.array([388.6725682632502, 6048.101178435452, 17332.199680774567,
                  216.03287586930986, 0.21676249605280398])
    cosmology = FlatLambdaCDM(H0=67.04, Om0=0.21479, Ob0=0.04895)
    model = 'Takahashi'

    # Test that inputs have correct shape and type
    assert isinstance(z, int) or isinstance(z, float),\
        " The redshift is not a float nor an integer"
    assert p.shape == k.shape,\
        " The input power spectrum does not have the same shape as wavenumber"
    assert isinstance(model, str),\
        " The model should be a string {\'Takahashi\', \'Smith\'}"
    if model != 'Takahashi' and model != 'Smith':
        print(" The model should be a string {\'Takahashi\', \'Smith\'}")

    # Test that output is a one-dimensional array
    nl_power = halofit(k, z, p, cosmology, model)
    assert nl_power.shape == k.shape,\
        " The power spectrum does not have the same shape as wavenumber"

    # Test against precomputed values
    precomputed_halo = np.array([3.88647217e+02, 6.04109819e+03,
                                 1.69587688e+04, 6.58984953e+02,
                                 1.11790397e+01])

    assert allclose(nl_power, precomputed_halo)
