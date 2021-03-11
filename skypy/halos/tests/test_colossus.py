import numpy as np
from scipy import integrate
from scipy.stats import kstest
import pytest
from skypy.halos._colossus import HAS_COLOSSUS


@pytest.mark.skipif(not HAS_COLOSSUS, reason='test requires colossus')
@pytest.mark.flaky
def test_colossus_mass_sampler():
    from astropy.cosmology import WMAP9
    from colossus.cosmology.cosmology import fromAstropy
    from colossus.lss import mass_function
    from skypy.halos.mass import colossus_mass_sampler
    m_min, m_max, size = 1e+12, 1e+16, 100
    sample = colossus_mass_sampler(redshift=0.1, model='despali16',
                                   mdef='500c', m_min=m_min, m_max=m_max,
                                   cosmology=WMAP9, sigma8=0.8200, ns=0.9608,
                                   size=size, resolution=1000)
    assert np.all(sample >= m_min)
    assert np.all(sample <= m_max)
    assert np.shape(sample) == (size,)
    fromAstropy(WMAP9, sigma8=0.8200, ns=0.9608)
    h0 = WMAP9.h
    m_h0 = np.logspace(np.log10(1e+12*h0), np.log10(1e+16*h0), 1000)
    dndm = mass_function.massFunction(m_h0, 0.1, mdef='500c', model='despali16',
                                      q_out='dndlnM', q_in='M')/m_h0
    m = m_h0/h0
    CDF = integrate.cumtrapz(dndm, (m), initial=0)
    CDF = CDF / CDF[-1]

    D, p = kstest(sample, lambda t: np.interp(t, m, CDF))

    assert p > 0.01, 'D = {}, p = {}'.format(D, p)


@pytest.mark.skipif(not HAS_COLOSSUS, reason='test requires colossus')
@pytest.mark.flaky
def test_colossus_mf_redshift():

    from skypy.halos._colossus import colossus_mf_redshift
    from astropy.cosmology import Planck18
    from astropy import units
    from scipy import integrate
    from colossus.lss.mass_function import massFunction

    # Parameters
    redshift = np.linspace(0., 2., 100)
    model, mdef = 'despali16', '500c'
    m_min, m_max = 1e+12, 1e+16
    sky_area = 1.0 * units.deg**2
    cosmology = Planck18
    sigma8, ns = 0.82, 0.96

    # Sample redshifts
    z_halo = colossus_mf_redshift(redshift, model, mdef, m_min, m_max, sky_area,
                                  cosmology, sigma8, ns, noise=False)

    # Integrate the mass function to get the number density of halos at each redshift
    def dndlnM(lnm, z):
        return massFunction(np.exp(lnm), z, 'M', 'dndlnM', mdef, model)
    lnmmin = np.log(m_min/cosmology.h)
    lnmmax = np.log(m_max/cosmology.h)
    density = [integrate.quad(dndlnM, lnmmin, lnmmax, args=(z))[0] for z in redshift]
    density = np.array(density) * np.power(cosmology.h, 3)

    # Integrate total number of halos for the given area of sky
    density *= (sky_area * cosmology.differential_comoving_volume(redshift)).to_value('Mpc3')
    n_halo = np.trapz(density, redshift, axis=-1)

    # make sure noise-free sample has right size
    assert np.isclose(len(z_halo), n_halo, atol=1.0)

    # Halo redshift CDF
    cdf = density  # same memory
    np.cumsum((density[1:]+density[:-1])/2*np.diff(redshift), out=cdf[1:])
    cdf[0] = 0
    cdf /= cdf[-1]

    # Check distribution of sampled halo redshifts
    D, p = kstest(z_halo, lambda z_: np.interp(z_, redshift, cdf))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

@pytest.mark.skipif(not HAS_COLOSSUS, reason='test requires colossus')
def test_colossus_mf():

    from skypy.halos._colossus import colossus_mf
    from astropy.cosmology import Planck18
    from astropy import units

    # redshift and mass distributions are tested separately
    # only test that output is consistent here

    z = np.linspace(0., 1., 100)
    model, mdef = 'despali16', '500c'
    m_min, m_max = 1e+12, 1e+16
    sky_area = 1.0 * units.deg**2
    cosmo = Planck18
    sigma8, ns = 0.82, 0.96
    z_halo, m_halo = colossus_mf(z, model, mdef, m_min, m_max, sky_area, cosmo, sigma8, ns)

    assert len(z_halo) == len(m_halo)
    assert np.all(z_halo >= np.min(z))
    assert np.all(z_halo <= np.max(z))
    assert np.all(m_halo >= m_min)
    assert np.all(m_halo <= m_max)
