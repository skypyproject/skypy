import numpy as np
import pytest
from scipy.stats import kstest


@pytest.mark.flaky
def test_schechter_lf_redshift():

    from skypy.galaxy.redshift import schechter_lf_redshift, redshifts_from_comoving_density
    from astropy.cosmology import FlatLambdaCDM
    from scipy.special import gamma, gammaincc

    # fix this cosmology
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    # parameters for the sampling
    z = np.linspace(1e-10, 2., 1000)
    M_star = -20
    phi_star = 1e-3
    alpha = -0.5
    m_lim = 30.
    fsky = 1/41253

    # sample redshifts
    z_gal = schechter_lf_redshift(z, M_star, phi_star, alpha, m_lim, fsky, cosmo, noise=False)

    # the absolute magnitude limit as function of redshift
    M_lim = m_lim - cosmo.distmod(z).value

    # lower limit of unscaled Schechter random variable
    x_min = 10.**(-0.4*(M_lim - M_star))

    # density with factor from upper incomplete gamma function
    density = phi_star*gamma(alpha+1)*gammaincc(alpha+1, x_min)

    # turn into galaxies/surface area
    density *= 4*np.pi*fsky*cosmo.differential_comoving_volume(z).to_value('Mpc3/sr')

    # integrate total number
    n_gal = np.trapz(density, z, axis=-1)

    # make sure noise-free sample has right size
    assert np.isclose(len(z_gal), n_gal, atol=1.0)

    # turn density into CDF
    cdf = density  # same memory
    np.cumsum((density[1:]+density[:-1])/2*np.diff(z), out=cdf[1:])
    cdf[0] = 0
    cdf /= cdf[-1]

    # check distribution of sample
    D, p = kstest(z_gal, lambda z_: np.interp(z_, z, cdf))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)


@pytest.mark.flaky
def test_redshifts_from_comoving_density():

    from skypy.galaxy.redshift import redshifts_from_comoving_density
    from astropy.cosmology import LambdaCDM

    # random cosmology
    H0 = np.random.uniform(50, 100)
    Om = np.random.uniform(0.1, 0.9)
    Ol = np.random.uniform(0.1, 0.9)
    cosmo = LambdaCDM(H0=H0, Om0=Om, Ode0=Ol)

    # fixed comoving density of Ngal galaxies total
    Ngal = 1000
    redshift = np.arange(0.0, 2.001, 0.1)
    density = Ngal/cosmo.comoving_volume(redshift[-1]).to_value('Mpc3')
    fsky = 1.0

    # sample galaxies over full sky without Poisson noise
    z_gal = redshifts_from_comoving_density(redshift, density, fsky, cosmo, noise=False)

    # make sure number of galaxies is correct (no noise)
    assert np.isclose(len(z_gal), Ngal, atol=1, rtol=0)

    # test the distribution of the sample against the analytical CDF
    V_max = cosmo.comoving_volume(redshift[-1]).to_value('Mpc3')
    D, p = kstest(z_gal, lambda z: cosmo.comoving_volume(z).to_value('Mpc3')/V_max)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)


@pytest.mark.flaky
def test_smail():
    from skypy.galaxy.redshift import smail

    # sample a single redshift
    rvs = smail(1.3, 2.0, 1.5)
    assert np.isscalar(rvs)

    # sample 10 reshifts
    rvs = smail(1.3, 2.0, 1.5, size=10)
    assert rvs.shape == (10,)

    # sample with implicit sizes
    zm, a, b = np.ones(5), np.ones((7, 5)), np.ones((13, 7, 5))
    rvs = smail(z_median=zm, alpha=a, beta=b)
    assert rvs.shape == np.broadcast(zm, a, b).shape

    # check sampling, for alpha=0, beta=1, the distribution is exponential
    D, p = kstest(smail(0.69315, 1e-100, 1., size=1000), 'expon')
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # check sampling, for beta=1, the distribution matches a gamma distribution
    D, p = kstest(smail(2.674, 2, 1, size=1000), 'gamma', args=(3,))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # check sampling, the distribution is a generalised gamma distribution
    D, p = kstest(smail(0.832555, 1, 2, size=1000), 'gengamma', args=(1, 2))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
