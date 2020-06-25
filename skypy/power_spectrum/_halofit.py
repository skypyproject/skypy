from astropy.utils import isiterable
from collections import namedtuple
from functools import partial
import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy import optimize


__all__ = [
   'HalofitParameters',
   'halofit',
   'halofit_smith',
   'halofit_takahashi',
   'halofit_bird',
]


HalofitParameters = namedtuple(
    'HalofitParameters',
    ['a', 'b', 'c', 'gamma', 'alpha', 'beta', 'mu', 'nu', 'fa', 'fb',
     'l', 'm', 'p', 'r', 's', 't'])

_smith_parameters = HalofitParameters(
    [0.1670, 0.7940, 1.6762, 1.8369, 1.4861, -0.6206, 0.0],
    [0.3084, 0.9466, 0.9463, -0.9400, 0.0],
    [0.3214, 0.6669, -0.2807, -0.0793],
    [0.2989, 0.8649, 0.1631],
    [-0.1452, 0.3700, 1.3884, 0.0],
    [0.0, 0.0, 0.3401, 0.9854, 0.8291, 0.0],
    [0.1908, -3.5442],
    [1.2857, 0.9589],
    [-0.0732, -0.1423, 0.0725],
    [-0.0307, -0.0585, 0.0743],
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

_takahashi_parameters = HalofitParameters(
    [0.2250, 0.9903, 2.3706, 2.8553, 1.5222, -0.6038, 0.1749],
    [0.5716, 0.5864, -0.5642, -1.5474, 0.2279],
    [0.8161, 2.0404, 0.3698, 0.5869],
    [-0.0843, 0.1971, 0.8460],
    [-0.1959, 1.3373, 6.0835, -5.5274],
    [0.3980, 1.2490, 0.3157, -0.7354, 2.0379, -0.1682],
    [0.0, -np.inf],
    [3.6902, 5.2105],
    [-0.0732, -0.1423, 0.0725],
    [-0.0307, -0.0585, 0.0743],
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

_bird_parameters = HalofitParameters(
    [0.1670, 0.7940, 1.6762, 1.8369, 1.4861, -0.6206, 0.0],
    [0.3084, 0.9466, 0.9463, -0.9400, 0.0],
    [0.3214, 0.6669, -0.2807, -0.0793],
    [0.2224, 1.18075, -0.6719],
    [-0.1452, 0.3700, 1.3884, 0.0],
    [0.0, 0.0, 0.3401, 0.9854, 0.8291, 0.0],
    [0.1908, -3.5442],
    [1.2857, 0.9589],
    [-0.0732, -0.1423, 0.0725],
    [-0.0307, -0.0585, 0.0743],
    2.080, 1.2e-3, 26.3, -6.49, 1.44, 12.4)


def halofit(wavenumber, redshift, linear_power_spectrum,
            cosmology, parameters):
    r'''Computation of the non-linear halo power spectrum.

    This function computes the non-linear halo power spectrum, as a function
    of redshift and wavenumbers, following [1]_, [2]_ and [3]_.

    Parameters
    ----------
    k : (nk,) array_like
        Input wavenumbers in units of [Mpc^-1].
    z : (nz,) array_like
        Input redshifts
    P : (nk, nz) array_like
        Linear power spectrum for given wavenumbers and redshifts [Mpc^3].
    cosmology : astropy.cosmology.Cosmology
                Cosmology object providing method for the evolution of
                omega_matter with redshift.
    parameters : HalofitParameters
                 namedtuple containing the free parameters of the model.

    Returns
    -------
    pknl : (nk, nz) array_like
           Non-linear halo power spectrum in units of [Mpc^3].

    References
    ----------
    .. [1] R. E. Smith it et al., VIRGO Consortium,
           Mon. Not. Roy. Astron. Soc. 341, 1311 (2003).
    .. [2] R. Takahashi, M. Sato, T. Nishimichi, A. Taruya and M. Oguri,
           Astrophys. J. 761, 152 (2012).
    .. [3] S. Bird, M. Viel and M. G. Haehnelt,
           Mon. Not. Roy. Astron. Soc. 420, 2551 (2012).

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> kvec = np.array([1.00000000e-04, 1.01000000e+01])
    >>> zvalue = 0.0
    >>> pvec = np.array([388.6725682632502, 0.21676249605280398])
    >>> cosmo = FlatLambdaCDM(H0=67.04, Om0=0.21479, Ob0=0.04895)
    >>> halofit(kvec, zvalue, pvec, cosmo, _takahashi_parameters)
    array([388.67064424,   0.72797614])
    '''

    # Manage shapes of input arrays
    return_shape = np.shape(linear_power_spectrum)
    redshift = np.atleast_1d(redshift)
    if np.ndim(linear_power_spectrum) == 1:
        linear_power_spectrum = linear_power_spectrum[:, np.newaxis]

    # Declaration of variables
    if isiterable(redshift):
        redshift = np.asarray(redshift)
    if isiterable(wavenumber):
        wavenumber = np.asarray(wavenumber)
    if isiterable(linear_power_spectrum):
        linear_power_spectrum = np.asarray(linear_power_spectrum)
    if np.any(redshift < 0):
        raise ValueError('Redshifts must be non-negative')
    if np.any(wavenumber <= 0):
        raise ValueError('Wavenumbers must be strictly positive')
    if np.any(linear_power_spectrum < 0):
        raise ValueError('Linear power spectrum must be non-negative')
    if not np.all(sorted(wavenumber) == wavenumber):
        raise ValueError('Wavenumbers must be provided in ascending order')

    # Redshift-dependent quantities from cosmology
    omega_m_z = cosmology.Om(redshift)[:, np.newaxis]
    omega_nu_z = cosmology.Onu(redshift)[:, np.newaxis]
    omega_de_z = cosmology.Ode(redshift)[:, np.newaxis]
    wp1_z = 1.0 + cosmology.w(redshift)[:, np.newaxis]
    ode_1pw_z = omega_de_z * wp1_z

    # Linear power spectrum interpolated at each redshift
    k2 = np.square(wavenumber)
    k3 = np.power(wavenumber, 3)
    dl2kz = (linear_power_spectrum.T * k3) / (2 * np.pi * np.pi)
    dl2k = [interpolate.interp1d(np.log(wavenumber), np.log(d)) for d in dl2kz]
    lnk_lo = np.log(wavenumber[0])
    lnk_up = np.log(wavenumber[-1])

    # Integrals required to evaluate Smith et al. 2003 equations C5, C7 & C8
    def integrand_kn(lnk, lnR, lnd, n):
        R2 = np.exp(2*lnR)
        k2 = np.exp(2*lnk)
        dl2 = np.exp(lnd(lnk))
        return dl2 * np.power(k2, n/2) * np.exp(-k2*R2)

    def integral_kn(lnR, lnd, n, lnk_lo, lnk_up):
        integrand = partial(integrand_kn, lnR=lnR, lnd=lnd, n=n)
        return integrate.quad(integrand, lnk_lo, lnk_up)[0]

    # Find root at which sigma^2(R) == 1.0 for each redshift
    # Smith et al. 2003 equation C5 & C6
    def log_sigma_squared(lnR):
        ik0 = [integral_kn(r, d, 0, lnk_lo, lnk_up) for r, d in zip(lnR, dl2k)]
        return np.log(ik0)
    guess = np.zeros(np.size(redshift))
    root = optimize.fsolve(log_sigma_squared, guess)
    R = np.exp(root)[:, np.newaxis]
    ksigma = 1.0 / R
    y = wavenumber / ksigma

    # Evaluate integrals at lnR = root for each redshift
    ik0 = [integral_kn(r, d, 0, lnk_lo, lnk_up) for r, d in zip(root, dl2k)]
    ik2 = [integral_kn(r, d, 2, lnk_lo, lnk_up) for r, d in zip(root, dl2k)]
    ik4 = [integral_kn(r, d, 4, lnk_lo, lnk_up) for r, d in zip(root, dl2k)]
    ik0 = np.asarray(ik0)[:, np.newaxis]
    ik2 = np.asarray(ik2)[:, np.newaxis]
    ik4 = np.asarray(ik4)[:, np.newaxis]

    # Effective spectral index neff and curvature C
    # Smith et al. 2003 equations C7 & C8
    neff = (2 * R * R * ik2 / ik0) - 3
    c = (4 * R * R / ik0) * (ik2 + R * R * (ik2 * ik2 / ik0 - ik4))

    # Smith et al. 2003 equations C9-C16
    # With higher order terms from Takahashi et al. 2012 equations A6-A13
    p = parameters
    an = np.power(10, np.polyval(p.a[:5], neff) + p.a[5]*c + p.a[6]*ode_1pw_z)
    bn = np.power(10, np.polyval(p.b[:3], neff) + p.b[3]*c + p.a[4]*ode_1pw_z)
    cn = np.power(10, np.polyval(p.c[:3], neff) + p.c[3]*c)
    gamman = np.polyval(p.gamma[:2], neff) + p.gamma[2]*c
    alphan = np.abs(np.polyval(p.alpha[:3], neff) + p.alpha[3]*c)
    betan = np.polyval(p.beta[:5], neff) + p.beta[5]*c
    mun = np.power(10, np.polyval(p.mu, neff))
    nun = np.power(10, np.polyval(p.nu, neff))

    # Smith et al. 2003 equations C17 & C18
    fa = np.power(omega_m_z, np.asarray(p.fa)[:, np.newaxis, np.newaxis])
    fb = np.power(omega_m_z, np.asarray(p.fb)[:, np.newaxis, np.newaxis])
    f = np.ones((3, np.size(redshift), 1))
    mask = omega_m_z != 1
    fraction = omega_de_z[mask] / (1.0 - omega_m_z[mask])
    f[:, mask] = fraction * fb[:, mask] + (1.0 - fraction) * fa[:, mask]

    # Massive neutrino terms; Bird et al. 2012 equations A6, A9 and A10
    fnu = omega_nu_z / omega_m_z
    Qnu = fnu * (p.l - p.t * (omega_m_z - 0.3)) / (1 + p.m * np.power(y, 3))
    dl2kz = dl2kz * (1 + (p.p * fnu * k2) / (1 + 1.5 * k2))
    betan = betan + fnu * (p.r + p.s * np.square(neff))

    # Two-halo term, Smith et al. 2003 equation C2
    fy = 0.25 * y + 0.125 * np.square(y)
    dq2 = dl2kz * (np.power(1+dl2kz, betan) / (1 + alphan*dl2kz)) * np.exp(-fy)

    # One-halo term, Smith et al. 2003 equations C3 and C4
    # With massive neutrino factor Q_nu, Bird et al. 2012 equation A7
    dh2p = an * np.power(y, 3 * f[0])\
        / (1.0 + bn * np.power(y, f[1]) + np.power(cn * f[2] * y, 3 - gamman))
    dh2 = (1 + Qnu) * dh2p / (1.0 + mun / y + nun / (y * y))

    # Halofit non-linear power spectrum, Smith et al. 2003 equation C1
    pknl = 2 * np.pi * np.pi * (dq2 + dh2) / k3

    return pknl.T.reshape(return_shape)


halofit_smith = partial(halofit, parameters=_smith_parameters)
halofit_takahashi = partial(halofit, parameters=_takahashi_parameters)
halofit_bird = partial(halofit, parameters=_bird_parameters)
