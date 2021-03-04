import numpy as np
import colossus as colossus
from colossus.lss import mass_function
from scipy import integrate
from scipy.stats import kstest
import pytest

@pytest.mark.flaky
def test_colossus_mass_sampler():
    from astropy.cosmology import WMAP9
    from skypy.halos.sampler import colossus_mass_sampler
    m_min = 1e+12
    sample = colossus_mass_sampler(redshift=0.1, model='despali16',
                                   mdef='500c', m_min=m_min, m_max=1e+16,
                                   cosmology=WMAP9, sigma8=0.8200, ns=0.9608,
                                   size=100, resolution=1000)
    assert np.all(sample >= m_min)
        
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
