''' This modules computes the angular size of galaxies from
their physical size.'''

import numpy as np
from astropy.utils import isiterable


def angular_size(physical_size, redshift, cosmology):
    '''Angular distance.
    This function transforms physical radius into angular distance, described
    in [1].

    Parameters
    ----------
    physical_size : numpy.ndarray
        Array of physical size of galaxies in units of [Mpc], with size nr.
    redshift : numpy.ndarray
        Array of redshifts, with size nz, at which to evaluate the angular
        diameter distance.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.

    Results
    -------
    angular_size : numpy.ndarray
        Array of angular distances in units of [Mpc^-1] with shape (nz,nr).

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmology = FlatLambdaCDM(H0=67.04, Om0=0.3183, Ob0=0.047745)
    >>> r = np.logspace(-3,2,num=1000)
    >>> da = angular_size(r,1,cosmology)
    >>> da[0]
    <Quantity 5.85990062e-07 1 / Mpc>

    References
    ----------
    .. [1] D. W. Hogg, (1999), astro-ph/9905116.
    '''

    r = physical_size
    z = redshift
    c = cosmology

    if isiterable(z):
        z = z.reshape(len(z), 1)

    angular_diameter = c.angular_diameter_distance(z)
    angular_size = r / angular_diameter

    return angular_size


def half_light_angular_size(magnitude, redshift, cosmology,
                            a_mu=1.0, b_mu=0.0, sigma_physical=1.0):
    '''Half-light angular size.
    This function transforms the half-light physical radius of galaxies,
    equation 3.14 in [1], into the half-light angular distance,
    equation 3.15 in [1].

    Parameters
    ----------
    magnitude : numpy.ndarray
        Array of galaxy magnitudes with shape nM.
    redshift : numpy.ndarray
        Array of redshifts, with shape nz, at which to evaluate the angular
        diameter distance.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.
    a_mu, b_mu : float
        Lognormal distribution parameters for the physical radius of galaxies,
        described in [1]. Default values are 1.0  and 0.0, respectively.
    sigma_physical : float
        Standard deviation of the lognormal distribution for the
        physical radius of galaxies. Default is 1.0.

    Results
    -------
    angular_size : numpy.ndarray
        Array of half-light angular distances in units of [Mpc-1]
        with shape (nz, nM).

    Examples
    --------
    import numpy as np
    from astropy.cosmology import FlatLambdaCDM
    cosmology = FlatLambdaCDM(H0=67.04, Om0=0.3183, Ob0=0.047745)
    da = half_light_angular_size(-21.7,1,cosmology)
    <Quantity 3.98088255e-14 1 / Mpc>

    References
    ----------
    .. [1] J. Herbel, T. Kacprzak, A. Amara, A. Refregier, C.Bruderer and
           A. Nicola, JCAP 1708, 035 (2017).
    '''

    M = magnitude
    z = redshift
    c = cosmology

    if isiterable(z):
        z = z.reshape(len(z), 1)

    mu_physical = a_mu * M + b_mu
    half_light_radius = np.random.lognormal(mu_physical, sigma_physical)
    angular_diameter = c.angular_diameter_distance(z)

    angular_size = half_light_radius / angular_diameter

    return angular_size
