import numpy as np
import pytest

healpy = pytest.importorskip('healpy')


def test_uniform_in_pixel_correctness():
    from skypy.position import uniform_in_pixel

    # 768 healpix pixels
    nside = 8
    npix = healpy.nside2npix(nside)

    ipix = np.random.randint(npix)

    size = 1000
    pos = uniform_in_pixel(nside, ipix, size=size)
    assert len(pos) == size

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
    size = 20
    theta, phi = np.empty(npix*size), np.empty(npix*size)
    for ipix in range(npix):
        pos = uniform_in_pixel(nside, ipix, size=size)
        assert len(pos) == size
        theta[ipix*size:(ipix+1)*size] = np.pi/2 - pos.dec.rad
        phi[ipix*size:(ipix+1)*size] = pos.ra.rad

    # test for uniform distribution
    D, p = kstest(theta, lambda t: (1 - np.cos(t))/2)
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
    D, p = kstest(phi, 'uniform', args=(0, 2*np.pi))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
