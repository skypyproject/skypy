''' This modules computes the angular size of galaxies from
their physical size.'''

import numpy as np
from astropy import units
from astropy.coordinates import Angle


def angular_size(radius, redshift, cosmology):
    '''Angular size of a galaxy.
    This function transforms physical radius into angular distance, described
    in [1].

    Parameters
    ----------
    radius : float
        Physical radius of galaxies in units of [kpc].
    redshift : float
        Redshifts at which to evaluate the angular diameter distance.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.

    Results
    -------
    angular_size : float
        Angular distances in units of [rad] for a given radius.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmology = FlatLambdaCDM(H0=67.04, Om0=0.3183, Ob0=0.047745)
    >>> angular_size(10 * units.kpc, 1, cosmology)
    0.0058599006191385255

    References
    ----------
    .. [1] D. W. Hogg, (1999), astro-ph/9905116.
    '''

    distance = cosmology.angular_diameter_distance(redshift)
    return np.arctan(physical_size / distance)

    return Angle(angular_size, unit=units.radian)


def linear_lognormal(magnitude, a_mu=1.0 * units.kpc, b_mu=0.0 * units.kpc,
                     sigma_physical=1.0 * units.kpc):
    '''Lognormal distribution with linear mean.
    This function provides a lognormal distribution for the physical size of
    galaxies with mean given by equation 3.14 in [1].

    Parameters
    ----------
    magnitude : float
        Galaxy magnitude at which evaluate the lognormal distribution.
    a_mu, b_mu : float
        Lognormal distribution parameters for the physical radius of galaxies,
        described in [1]. Default values are 1.0  and 0.0 [kpc], respectively.
    sigma_physical : float
        Standard deviation of the lognormal distribution for the
        physical radius of galaxies. Default is 1.0 [kpc].

    Results
    -------
    radius : numpy.ndarray
        Physical distance for a given galaxy with a given magnitude,
        in units of [kpc].

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> linear_lognormal(26)
    159497221748.5613

    References
    ----------
    .. [1] J. Herbel, T. Kacprzak, A. Amara, A. Refregier, C.Bruderer and
           A. Nicola, JCAP 1708, 035 (2017).
    '''

    mu_physical = a_mu * magnitude + b_mu
    sigma_physical = sigma_physical.to(mu_physical.unit)
    physical_size = scipy.stats.lognorm.rvs(mu_physical, sigma_physical, size=size)
    return physical_size * mu_physical.unit

    return radius
