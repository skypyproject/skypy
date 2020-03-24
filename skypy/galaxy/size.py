''' This modules computes the angular size of galaxies from
their physical size.'''

import numpy as np
from astropy import units


def angular_size(physical_size, redshift, cosmology):
    '''Angular size of a galaxy.
    This function transforms physical radius into angular distance, described
    in [1].

    Parameters
    ----------
    physical_size : astropy.Quantity
        Physical radius of galaxies in units of length.
    redshift : float
        Redshifts at which to evaluate the angular diameter distance.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.

    Results
    -------
    angular_size : astropy.Quantity
        Angular distances in units of [rad] for a given radius.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmology = FlatLambdaCDM(H0=67.04, Om0=0.3183, Ob0=0.047745)
    >>> size = 10.0 * units.kpc
    >>> angular_size(size, 1, cosmology)
    <Quantity 5.85990062e-06 rad>

    References
    ----------
    .. [1] D. W. Hogg, (1999), astro-ph/9905116.
    '''

    distance = cosmology.angular_diameter_distance(redshift)
    size_physical = units.Quantity(physical_size).to(distance.unit)

    angular_size = np.arctan(size_physical / distance)

    return angular_size


def linear_lognormal(magnitude, a_mu, b_mu, sigma_physical, size=None):
    '''Lognormal distribution with linear mean.
    This function provides a lognormal distribution for the physical size of
    galaxies with mean given by equation 3.14 in [1].

    Parameters
    ----------
    magnitude : float
        Galaxy magnitude at which evaluate the lognormal distribution.
    a_mu, b_mu : astropy.Quantity
        Lognormal distribution parameters for the physical radius of galaxies
        in units of length, described in [1].
    sigma_physical : astropy.Quantity
        Standard deviation of the lognormal distribution for the
        physical radius of galaxies in units of length.
    size : int or tuple of ints, optional.
        Output shape. If the given shape is, e.g., (m, n, k),
        then m * n * k samples are drawn. If size is None (default),
        a single value is returned if mean and sigma are both scalars.
        Otherwise, np.broadcast(mean, sigma).size samples are drawn.

    Results
    -------
    physical_size : numpy.ndarray or astropy.Quantity
        Physical distance for a given galaxy with a given magnitude,
        in the same length units as input parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units
    >>> np.random.seed(12345)
    >>> magnitude = 26.0
    >>> a_mu = -0.25 * units.kpc
    >>> b_mu = 1.0 * units.kpc
    >>> sigma = 0.5 * units.kpc
    >>> linear_lognormal(magnitude, a_mu, b_mu, sigma)
    <Quantity 0.00368917 kpc>

    References
    ----------
    .. [1] J. Herbel, T. Kacprzak, A. Amara, A. Refregier, C.Bruderer and
           A. Nicola, JCAP 1708, 035 (2017).
    '''

    mu_physical = units.Quantity(a_mu * magnitude + b_mu)
    sigma_physical = units.Quantity(sigma_physical).to(mu_physical.unit)
    mu_value = mu_physical.value
    sigma_value = sigma_physical.value

    size_value = np.random.lognormal(mu_value, sigma_value, size=size)
    size_physical = size_value * mu_physical.unit

    return size_physical
