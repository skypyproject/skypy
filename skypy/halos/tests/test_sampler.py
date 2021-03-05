import numpy as np
from scipy import integrate
from scipy.stats import kstest
import pytest
from skypy.halos.sampler import HAS_COLOSSUS


@pytest.mark.skipif(not HAS_COLOSSUS, reason='test requires colossus')
@pytest.mark.flaky
def test_colossus_mass_sampler():
    from astropy.cosmology import WMAP9
    import colossus as colossus
    from colossus.lss import mass_function
    from skypy.halos.sampler import colossus_mass_sampler
    m_min, m_max, size = 1e+12, 1e+16, 100
    sample = colossus_mass_sampler(redshift=0.1, model='despali16',
                                   mdef='500c', m_min=m_min, m_max=m_max,
                                   cosmology=WMAP9, sigma8=0.8200, ns=0.9608,
                                   size=size, resolution=1000)
    assert np.all(sample >= m_min)
    assert np.all(sample <= m_max)
    assert np.shape(sample) == (size,)
    cosmo = colossus.cosmology.cosmology.fromAstropy(WMAP9, sigma8=0.8200,
                                                     ns=0.9608, name='my_cosmo')
    h0 = cosmo.h
    m_h0 = np.logspace(np.log10(1e+12*h0), np.log10(1e+16*h0), 1000)
    dndm = mass_function.massFunction(m_h0, 0.1, mdef='500c', model='despali16',
                                      q_out='dndlnM', q_in='M')/m_h0
    m = m_h0/h0
    CDF = integrate.cumtrapz(dndm, (m), initial=0)
    CDF = CDF / CDF[-1]

    D, p = kstest(sample, lambda t: np.interp(t, m, CDF))

    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
