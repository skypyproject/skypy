import numpy as np
import pytest
from scipy.stats import kstest


@pytest.mark.flaky
def test_schechter_lf_magnitude():
    from skypy.galaxies.luminosity import schechter_lf_magnitude
    from astropy.cosmology import default_cosmology
    import pytest

    # use default cosmology
    cosmo = default_cosmology.get()

    # Schechter function parameters for tests
    M_star = -20.5
    alpha = -1.3

    # sample 1000 galaxies at a fixed redshift of 1.0
    z = np.repeat(1.0, 1000)
    M = schechter_lf_magnitude(z, M_star, alpha, 30., cosmo)

    # get the distribution function
    log10_x_min = -0.4*(30. - cosmo.distmod(1.0).value - M_star)
    x = np.logspace(log10_x_min, log10_x_min + 3, 1000)
    pdf = x**(alpha+1)*np.exp(-x)
    cdf = np.concatenate([[0.], np.cumsum((pdf[1:]+pdf[:-1])/2*np.diff(np.log(x)))])
    cdf /= cdf[-1]

    # test the samples against the CDF
    D, p = kstest(10.**(-0.4*(M - M_star)), lambda t: np.interp(t, x, cdf))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)

    # test for 1000 galaxies with Pareto redshift distribution
    z = np.random.pareto(3., size=1000)

    # for scalar parameters, sample galaxies with magnitude limit of 30
    M = schechter_lf_magnitude(z, M_star, alpha, 30., cosmo)

    # check that the output has the correct shape
    assert np.shape(M) == (1000,)

    # make sure magnitude limit was respected
    M_lim = 30. - cosmo.distmod(z).value
    assert np.all(M <= M_lim)

    # sample with array for alpha
    # not implemented at the moment
    with pytest.raises(NotImplementedError):
        M = schechter_lf_magnitude(z, M_star, np.broadcast_to(alpha, z.shape), 30., cosmo)

    # sample with an explicit size
    schechter_lf_magnitude(1.0, M_star, alpha, 30., cosmo, size=100)
