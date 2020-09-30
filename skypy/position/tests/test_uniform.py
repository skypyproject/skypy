import numpy as np
import pytest
from scipy.stats import kstest


@pytest.mark.flaky
def test_uniform_around():
    from skypy.position import uniform_around
    from astropy.coordinates import SkyCoord
    from astropy import units

    # random position on the sphere
    ra = np.random.uniform(0, 2*np.pi)
    dec = np.pi/2 - np.arccos(np.random.uniform(-1, 1))
    centre = SkyCoord(ra, dec, unit=units.rad)

    # random maximum separation
    theta_max = np.random.uniform(0, np.pi)

    # area from opening angle = separation
    area = 2*np.pi*(1 - np.cos(theta_max))*units.sr

    # sample 1000 random positions
    coords = uniform_around(centre, area, 1000)

    # make sure sample respects size
    assert len(coords) == 1000

    # compute separations and position angles from centre
    theta = centre.separation(coords).radian
    phi = centre.position_angle(coords).radian

    # test for uniform distribution
    D, p = kstest(theta, lambda t: (1 - np.cos(t))/(1 - np.cos(theta_max)))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
    D, p = kstest(phi, 'uniform', args=(0., 2*np.pi))
    assert p > 0.01, 'D = {}, p = {}'.format(D, p)
