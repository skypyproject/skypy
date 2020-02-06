''' This module computes the non-linear halo power spectrum as a function of
    redshift and wavenumbers.
    '''

import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from scipy import optimize


def halofit(wavenumber, redshift, linear_power_spectrum,
            cosmology, model='Takahashi'):
    """ This function computes the non-linear halo power spectrum, as a function
        of redshift and wavenumbers.
        One can choose from two different models: 'Takahashi' or 'Smith',
        described in [1] and [2], respectively.

        Parameters
        ----------
        k : array_like
            Imput wavenumbers in units of [Mpc^-1].
        z : integer or float
            Input redshifts.
        P : array_like
            Linear power spectrum for a single redshift [Mpc^3].
        cosmology : array_like
                    Astropy-like cosmology.
        model : string
                'Takahashi' (default model),
                'Smith'.

        Returns
        -------
        pknl : array_like
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
        >>> halofit(kvec, zvalue, pvec, cosmo)[0]
        388.6629999679634
        """
    # Declaration of variables
    z = redshift
    k = wavenumber
    P = linear_power_spectrum

    # Cosmology
    omega_m_z = cosmology.Om(z)

    # Linear power spectrum
    k2 = k * k
    k3 = k2 * k
    pi2 = np.pi * np.pi
    dl2_kz = (P * k3) / (2 * pi2)
    dl2k = interpolate.interp1d(k, dl2_kz)

    # Equation A4 sigma^2(R)
    def sigma_squared(R):
        ''' Equation A4, sigma^2(R)
            '''
        R2 = R * R

        def integrand(x):
            return (dl2k(x) * np.exp(-x * x * R2)) / x

        integrand = quad(integrand, k[0], k[-1], limit=100)[0]

        return integrand

    # First and second derivatives of sigma^2(R), equation A5
    def dln_sigma_squared(R):
        ''' First derivative of sigma^2(R), c.f.  neff in equation A5
            '''
        R2 = R * R

        def integrand(x):
            return dl2k(x) * np.exp(-x * x * R2) * x

        res = quad(integrand, k[0], k[-1], limit=100)[0]
        res = -2 * R2 * res / sigma_squared(R)
        return res

    def d2ln_sigma_squared(R):
        ''' Second derivative of sigma^2(R), c.f.  C in equation A5
            '''
        R2 = R * R
        R4 = R2 * R2

        term1 = 2 * dln_sigma_squared(R)
        term2 = - np.power(dln_sigma_squared(R), 2)

        def integrand(x):
            return dl2k(x) * np.exp(-x * x * R2) * np.power(x, 3)
        integral3 = quad(integrand, k[0], k[-1], limit=100)[0]
        term3 = 4 * R4 * integral3 / sigma_squared(R)

        res = term1 + term2 + term3
        return res

    # Find root at which sigma^2(R) == 1.0
    def equation(R):
        equation = sigma_squared(R) - 1.0
        return equation

    Rroot = optimize.fsolve(equation, 2.0)[0]

    # Evaluation at R = root
    s2r = dln_sigma_squared(Rroot)
    s3r = d2ln_sigma_squared(Rroot)
    ksigma = 1.0 / Rroot

    # Effective spectral index neff and curvature C, equation A5
    neff = (- 3 - s2r)
    neff2 = np.square(neff)
    neff3 = neff2 * neff
    neff4 = neff3 * neff
    c = - s3r

    # Coefficients
    if model == 'Takahashi':
        # Equations A6-13
        anv = [1.5222, 2.8553, 2.3706, 0.9903, 0.2250, -0.6038]
        bnv = [-0.5642, 0.5864, 0.5716, -1.5474]
        cnv = [0.3698, 2.0404, 0.8161, 0.5869]
        gammanv = [0.1971, -0.0843, 0.8460]
        alphanv = [6.0835, 1.3373, -0.1959, -5.5274]
        betanv = [2.0379, -0.7354, 0.3157, 1.2490, 0.3980, -0.1682]
        munv = [-np.inf, 0.0]
        nunv = [5.2105, 3.6902]
    elif model == 'Smith':
        # Equations C9-16
        anv = [1.4861, 1.8369, 1.6762, 0.7940, 0.1670, -0.6206]
        bnv = [0.9463, 0.9466, 0.3084, -0.9400]
        cnv = [-0.2807, 0.6669, 0.3214, -0.0793]
        gammanv = [0.8649, 0.2989, 0.1631]
        alphanv = [1.3884, 0.3700, -0.1452, 0.0]
        betanv = [0.8291, 0.9854, 0.3401, 0.0, 0.0, 0.0]
        munv = [-3.5442, 0.1908]
        nunv = [0.9589, 1.2857]
    else:
        anv = [-np.inf, 0.0, 0.0, 0.0, 0.0, 0.0]
        bnv = [-np.inf, 0.0, 0.0, 0.0]
        cnv = [-np.inf, 0.0, 0.0, 0.0]
        gammanv = np.zeros(3)
        alphanv = np.zeros(4)
        betanv = np.zeros(6)
        munv = [-np.inf, 0.0]
        nunv = [-np.inf, 0.0]

    # Parameters
    an = np.power(10, anv[0] + anv[1] * neff + anv[2] * neff2 + anv[3] * neff3
                  + anv[4] * neff4 + anv[5] * c)
    bn = np.power(10, bnv[0] + bnv[1] * neff + bnv[2] * neff2 + bnv[3] * c)
    cn = np.power(10, cnv[0] + cnv[1] * neff + cnv[2] * neff2 + cnv[3] * c)
    gamman = gammanv[0] + gammanv[1] * neff + gammanv[2] * c
    alphan = np.abs(alphanv[0] + alphanv[1] * neff + alphanv[2] * neff2
                    + alphanv[3] * c)
    betan = betanv[0] + betanv[1] * neff + betanv[2] * neff2\
        + betanv[3] * neff3 + betanv[4] * neff4 + betanv[5] * c
    mun = np.power(10, munv[0] + munv[1] * neff)
    nun = np.power(10, nunv[0] + nunv[1] * neff)

    # Equation A14
    f1 = np.power(omega_m_z, -0.0307)
    f2 = np.power(omega_m_z, -0.0585)
    f3 = np.power(omega_m_z,  0.0743)

    # Equations A1, A2 & A3
    y = k / ksigma
    y2 = y * y
    fy = 0.25 * y + 0.125 * np.square(y)

    # Two-halo contribution
    dq2 = dl2_kz * (np.power(1 + dl2_kz, betan) / (1 + alphan * dl2_kz))\
        * np.exp(-fy)

    # One-halo contribution dh2 and its derivative dh2p
    dh2p = an * np.power(y, 3 * f1)\
        / (1.0 + bn * np.power(y, f2) + np.power(cn * f3 * y, 3 - gamman))
    dh2 = dh2p / (1.0 + mun / y + nun / y2)

    # halo power spectrum
    pknl = 2 * pi2 * (dq2 + dh2) / k3

    return pknl.T
