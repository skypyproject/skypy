import numpy as np
import pytest

healpy = pytest.importorskip('healpy')


def test_uniform_in_pixel_correctness():
    from skypy.position import uniform_in_pixel

    # 768 healpix pixels
    nside = 8
    npix = healpy.nside2npix(nside)

    ipix = np.random.randint(npix)

    pos = uniform_in_pixel(nside, ipix, size=1000)
    assert len(pos) == 1000

    lon, lat = pos.ra.deg, pos.dec.deg
    apix = healpy.ang2pix(nside, lon, lat, lonlat=True)
    np.testing.assert_array_equal(apix, ipix)


@pytest.mark.flaky
def test_uniform_in_pixel_distribution():
    from scipy.stats import kstest

    from skypy.position import uniform_in_pixel

    # 48 healpix pixels
    nside = 2
    npix = healpy.nside2npix(nside)

    # sample entire sky with 20 positions per pixel
    theta, phi = np.empty(npix*20), np.empty(npix*20)
    for ipix in range(npix):
        pos = uniform_in_pixel(nside, ipix, size=20)
        assert len(pos) == 20
        theta[ipix*20:(ipix+1)*20] = np.pi/2 - pos.dec.rad
        phi[ipix*20:(ipix+1)*20] = pos.ra.rad

    # test for uniform distribution
    D, p = kstest(theta, lambda t: (1 - np.cos(t))/2)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
    D, p = kstest(phi, 'uniform', args=(0, 2*np.pi))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
