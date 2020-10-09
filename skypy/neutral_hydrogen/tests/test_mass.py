import numpy as np
import pytest

from astropy.cosmology import FlatLambdaCDM
from astropy import units

pr_data = np.asarray([0.00000000e+00, 8.08547768e-45, 6.40736312e+09, 3.66714636e+09,
                      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]) * units.Msun


def test_pr_halo_model():

    from skypy.neutral_hydrogen.mass import pr_halo_model

    cosmology = FlatLambdaCDM(Om0=1.0, H0=70.0, Ob0=0.045)

    m_halo = np.logspace(8, 15, 8) * units.Msun
    v_halo = (96.6 * (units.km / units.s)) * m_halo / (1.e11 * units.Msun)

    m_hone = pr_halo_model(m_halo, v_halo, cosmology)

    # test scalar output
    assert np.isscalar(pr_halo_model(m_halo[0], v_halo[0], cosmology).value)

    # test array output
    assert pr_halo_model(m_halo, v_halo, cosmology).value.shape == (8,)

    # test pre-computed example
    assert units.allclose(pr_halo_model(m_halo, v_halo, cosmology), pr_data)
