import numpy as np


def eisenstein_hu(wavenumber, A_s, n_s, cosmology, kwmap=0.02, wiggle=True):
    """ Eisenstein-Hu fitting function for the linear matter
    power spectrum with (or without) acoustic osscilations described in [1],
    [2].

    Parameters
    ----------
    wavenumber : array_like
        Array of wavenumbers of length nk in units of h Mpc^-1 at which to
        evaluate the linear matter power spectrum.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing omega_matter, omega_baryon, Hubble parameter
        and CMB temperature in the present day
    wiggle : bool
        Boolean flag to specify the use of the Baryonic Acoustic Oscillations
        wiggle approximation.
    A_s, n_s: float
        Amplitude and spectral index of primordial scalar fluctuations.
    kwmap : float
        WMAP normalization for the amplitude of primordial scalar fluctuations,
        as described in [3].

    Returns
    -------
    power_spectrum : array_like
        Array of values for the linear matter power spectrum evaluated at the
        input wavenumbers for the given primordial power spectrum parameters,
        cosmology, kwmap normalization and choice of wiggle boolean flag.

    References
    ----------
        [1] Eisenstein D. J., Hu W., ApJ, 496, 605 (1998)
        [2] Eisenstein D. J., Hu W., ApJ, 511, 5 (1999)
        [3] Komatsu et al., ApJS, 180, 330 (2009)
    """

    def compute_pk(wavenumber, A_s, n_s, cosmology, kwmap, wiggle=True):
        om0 = cosmology.Om0
        ob0 = cosmology.Ob0
        h0 = cosmology.H0.value / 100
        Tcmb0 = cosmology.Tcmb0.value
        if wiggle:
            trans = eisensteinhu_withwiggle(wavenumber * h0, om0 * h0**2,
                                            ob0 * h0**2, ob0 / om0, Tcmb0)
        else:
            trans = eisensteinhu_nowiggle(wavenumber * h0, om0 * h0**2,
                                          ob0 / om0, Tcmb0)
        # Eq [74] in [1]
        pk = A_s * (2.0 * wavenumber**2 * 2998.0**2 / 5.0 / om0)**2 * \
            trans**2 * (wavenumber * h0 / kwmap)**(n_s - 1.0) * 2.0 * \
            np.pi**2 / wavenumber**3

        return pk

    def eisensteinhu_nowiggle(ak, omegamh2, fb, tcmb=2.7255):
        alpha = 1.0 - 0.328 * np.log(431.0 * omegamh2) * fb + 0.38 * \
            np.log(22.3 * omegamh2) * fb**2
        sound = 44.5 * np.log(9.83 / omegamh2) / \
            np.sqrt(1.0 + 10.0 * (fb * omegamh2)**(0.75))
        shape = omegamh2 * (alpha + (1.0 - alpha) /
                            (1.0 + (0.43 * ak * sound)**4))
        aq = ak * (tcmb / 2.7)**2 / shape
        T = np.log(2.0 * np.exp(1) + 1.8 * aq) / \
            (np.log(2.0 * np.exp(1) + 1.8 * aq) +
             (14.2 + 731 / (1 + 62.5 * aq)) * aq * aq)

        return T

    def eisensteinhu_withwiggle(ak, Omh2, Obh2, f_baryon, tcmb=2.7255):
        # redshift and wavenumber equality
        k_eq = 7.46e-2 * Omh2 * (tcmb / 2.7)**-2
        z_eq = 2.5e4 * Omh2 * (tcmb / 2.7)**-4

        # sound horizon and k_silk
        z_drag_b1 = 0.313 * Omh2**-0.419 * (1 + 0.607 * Omh2**0.674)
        z_drag_b2 = 0.238 * Omh2**0.223
        z_drag = 1291 * Omh2**0.251 / (1 + 0.659 * Omh2**0.828) \
            * (1 + z_drag_b1 * Obh2**z_drag_b2)

        r_drag = 31.5 * Obh2 * (tcmb / 2.7)**-4 * (1000. / z_drag)
        r_eq = 31.5 * Obh2 * (tcmb / 2.7)**-4 * (1000. / z_eq)

        sound_horizon = 2 / (3 * k_eq) * np.sqrt(6 / r_eq) * \
            np.log((np.sqrt(1 + r_drag) + np.sqrt(r_drag + r_eq)) /
                   (1 + np.sqrt(r_eq)))
        k_silk = 1.6 * Obh2**0.52 * Omh2**0.73 * (1 + (10.4 * Omh2)**-0.95)

        # alpha c
        alpha_c_a1 = (46.9 * Omh2)**0.670 * (1 + (32.1 * Omh2)**-0.532)
        alpha_c_a2 = (12.0 * Omh2)**0.424 * (1 + (45.0 * Omh2)**-0.582)
        alpha_c = alpha_c_a1 ** -f_baryon * alpha_c_a2 ** (-f_baryon**3)

        # beta_c
        beta_c_b1 = 0.944 / (1 + (458 * Omh2)**-0.708)
        beta_c_b2 = (0.395 * Omh2)**-0.0266
        beta_c = 1 / (1 + beta_c_b1 * ((1 - f_baryon)**beta_c_b2 - 1))

        y = (1 + z_eq) / (1 + z_drag)
        alpha_b_G = y * (-6 * np.sqrt(1 + y) + (2 + 3 * y)
                         * np.log((np.sqrt(1 + y) + 1) / (np.sqrt(1 + y) - 1)))
        alpha_b = 2.07 * k_eq * sound_horizon * (1 + r_drag)**-0.75 * alpha_b_G

        beta_node = 8.41 * Omh2 ** 0.435
        beta_b = 0.5 + f_baryon + (3 - 2 * f_baryon) * np.sqrt((17.2 * Omh2)**2
                                                               + 1)

        q = ak / (13.41 * k_eq)
        ks = ak * sound_horizon

        T_c_ln_beta = np.log(np.e + 1.8 * beta_c * q)
        T_c_ln_nobeta = np.log(np.e + 1.8 * q)
        T_c_C_alpha = 14.2 / alpha_c + 386. / (1 + 69.9 * q ** 1.08)
        T_c_C_noalpha = 14.2 + 386. / (1 + 69.9 * q ** 1.08)

        T_c_f = 1 / (1 + (ks / 5.4) ** 4)

        def f(a, b):
            return a / (a + b * q**2)

        T_c = T_c_f * f(T_c_ln_beta, T_c_C_noalpha) + \
            (1 - T_c_f) * f(T_c_ln_beta, T_c_C_alpha)

        s_tilde = sound_horizon * (1 + (beta_node / ks)**3)**(-1 / 3)
        ks_tilde = ak * s_tilde

        T_b_T0 = f(T_c_ln_nobeta, T_c_C_noalpha)
        T_b_1 = T_b_T0 / (1 + (ks / 5.2)**2)
        T_b_2 = alpha_b / (1 + (beta_b / ks)**3) * np.exp(-(ak / k_silk)**1.4)
        T_b = np.sinc(ks_tilde / np.pi) * (T_b_1 + T_b_2)

        T = f_baryon * T_b + (1 - f_baryon) * T_c

        return T

    power_spectrum = compute_pk(wavenumber, A_s, n_s, cosmology, kwmap, wiggle)
    return power_spectrum
