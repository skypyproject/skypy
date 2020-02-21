''' This module computes the non-linear halo power spectrum as a function of
    redshift and wavenumbers.
    '''

from astropy.utils import isiterable
from collections import namedtuple
from functools import partial
import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy import optimize

_HalofitParameters = namedtuple(
    'HalofitParameters',
    ['a', 'b', 'c', 'gamma', 'alpha', 'beta', 'mu', 'nu'])

_smith_parameters = _HalofitParameters(
    [1.4861, 1.8369, 1.6762, 0.7940, 0.1670, -0.6206],
    [0.9463, 0.9466, 0.3084, -0.9400],
    [-0.2807, 0.6669, 0.3214, -0.0793],
    [0.8649, 0.2989, 0.1631],
    [1.3884, 0.3700, -0.1452, 0.0],
    [0.8291, 0.9854, 0.3401, 0.0, 0.0, 0.0],
    [-3.5442, 0.1908],
    [0.9589, 1.2857])

_takahashi_parameters = _HalofitParameters(
    [1.5222, 2.8553, 2.3706, 0.9903, 0.2250, -0.6038],
    [-0.5642, 0.5864, 0.5716, -1.5474],
    [0.3698, 2.0404, 0.8161, 0.5869],
    [0.1971, -0.0843, 0.8460],
    [6.0835, 1.3373, -0.1959, -5.5274],
    [2.0379, -0.7354, 0.3157, 1.2490, 0.3980, -0.1682],
    [-np.inf, 0.0],
    [5.2105, 3.6902])

_halofit_parameters = {
    'Smith': _smith_parameters,
    'Takahashi': _takahashi_parameters}


def halofit(wavenumber, redshift, linear_power_spectrum,
            cosmology, model='Takahashi'):
    """Computation of the non-linear halo power spectrum.
    This function computes the non-linear halo power spectrum, as a function
    of redshift and wavenumbers.
    One can choose from two different models: 'Takahashi' or 'Smith',
    described in [1] and [2], respectively.

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
    model : string
            'Takahashi' (default model),
            'Smith'.

    Returns
    -------
    pknl : (nk, nz) array_like
           Non-linear halo power spectrum, described in [1] or [2], in
           units of [Mpc^3].

    References
    ----------
        [1] R. Takahashi, M. Sato, T. Nishimichi, A. Taruya and M. Oguri,
            Astrophys. J. 761, 152 (2012).
        [2] R. E. Smith it et al., VIRGO Consortium,
            Mon. Not. Roy. Astron. Soc. 341, 1311 (2003).

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> kvec = np.array([1.00000000e-04, 1.01000000e+01])
    >>> zvalue = 0.0
    >>> pvec = np.array([388.6725682632502, 0.21676249605280398])
    >>> cosmo = FlatLambdaCDM(H0=67.04, Om0=0.21479, Ob0=0.04895)
    >>> halofit(kvec, zvalue, pvec, cosmo)
    array([388.67064424,   0.72797614])
    """

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

    # Cosmology
    omega_m_z = cosmology.Om(redshift)[:, np.newaxis]

    # Linear power spectrum
    k3 = np.power(wavenumber, 3)
    dl2kz = (linear_power_spectrum.T * k3) / (2 * np.pi * np.pi)
    dl2k = [interpolate.interp1d(np.log(wavenumber), np.log(d)) for d in dl2kz]
    low = np.log(wavenumber[0])
    hi = np.log(wavenumber[-1])

    # Integrals required to evaluate A4 and A5
    def integrand_kn(lnk, lnR, lnd, n):
        R2 = np.exp(2*lnR)
        k2 = np.exp(2*lnk)
        dl2 = np.exp(lnd(lnk))
        return dl2 * np.power(k2, n/2) * np.exp(-k2*R2)

    def integral_kn(lnR, lnd, n, low, hi):
        integrand = partial(integrand_kn, lnR=lnR, lnd=lnd, n=n)
        return integrate.quad(integrand, low, hi)[0]

    # Find root at which sigma^2(R) == 1.0 for each redshift, equation A4
    def log_sigma_squared(lnR):
        ik0 = [integral_kn(r, d, 0, low, hi) for r, d in zip(lnR, dl2k)]
        return np.log(ik0)
    guess = np.zeros(np.size(redshift))
    root = optimize.fsolve(log_sigma_squared, guess)
    R = np.exp(root)[:, np.newaxis]
    ksigma = 1.0 / R

    # Evaluation at lnR = root
    ik0 = [integral_kn(r, d, 0, low, hi) for r, d in zip(root, dl2k)]
    ik2 = [integral_kn(r, d, 2, low, hi) for r, d in zip(root, dl2k)]
    ik4 = [integral_kn(r, d, 4, low, hi) for r, d in zip(root, dl2k)]
    ik0 = np.array(ik0)[:, np.newaxis]
    ik2 = np.array(ik2)[:, np.newaxis]
    ik4 = np.array(ik4)[:, np.newaxis]

    # Effective spectral index neff and curvature C, equation A5
    neff = (2 * R * R * ik2 / ik0) - 3
    neff2 = np.square(neff)
    neff3 = neff2 * neff
    neff4 = neff3 * neff
    c = (4 * R * R / ik0) * (ik2 + R * R * (ik2 * ik2 / ik0 - ik4))

    # Equations A6-A14
    p = _halofit_parameters[model]
    an = np.power(10, p.a[0] + p.a[1] * neff + p.a[2] * neff2 + p.a[3] * neff3
                  + p.a[4] * neff4 + p.a[5] * c)
    bn = np.power(10, p.b[0] + p.b[1] * neff + p.b[2] * neff2 + p.b[3] * c)
    cn = np.power(10, p.c[0] + p.c[1] * neff + p.c[2] * neff2 + p.c[3] * c)
    gamman = p.gamma[0] + p.gamma[1] * neff + p.gamma[2] * c
    alphan = np.abs(p.alpha[0] + p.alpha[1] * neff + p.alpha[2] * neff2
                    + p.alpha[3] * c)
    betan = p.beta[0] + p.beta[1] * neff + p.beta[2] * neff2\
        + p.beta[3] * neff3 + p.beta[4] * neff4 + p.beta[5] * c
    mun = np.power(10, p.mu[0] + p.mu[1] * neff)
    nun = np.power(10, p.nu[0] + p.nu[1] * neff)
    f1 = np.power(omega_m_z, -0.0307)
    f2 = np.power(omega_m_z, -0.0585)
    f3 = np.power(omega_m_z,  0.0743)

    # Two-halo term, equation A2
    y = wavenumber / ksigma
    fy = 0.25 * y + 0.125 * np.square(y)
    dq2 = dl2kz * (np.power(1+dl2kz, betan) / (1 + alphan*dl2kz)) * np.exp(-fy)

    # One-halo term, equation A3
    dh2p = an * np.power(y, 3 * f1)\
        / (1.0 + bn * np.power(y, f2) + np.power(cn * f3 * y, 3 - gamman))
    dh2 = dh2p / (1.0 + mun / y + nun / (y * y))

    # Halofit non-linear power spectrum, equation A1
    pknl = 2 * np.pi * np.pi * (dq2 + dh2) / k3

    return pknl.T.reshape(return_shape)
