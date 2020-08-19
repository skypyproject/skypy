import numpy as np
from astropy import units as u


__all__ = [
    'classy',
]


def classy(wavenumber, redshift, cosmology, A_s, n_s):
    """ Return the CLASS computation of the linear matter power spectrum, on a
    two dimensional grid of wavenumber and redshift

    Parameters
    ----------
    wavenumber : (nk,) array_like
        Array of wavenumbers in units of [Mpc^-1] at which to
        evaluate the linear matter power spectrum.
    redshift : (nz,) array_like
        Array of redshifts at which to evaluate the linear matter power
        spectrum.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing omega_matter, omega_baryon, Hubble
        parameter and CMB temperature in the present day
    A_s : float
        Cosmology parameter, amplitude normalisation of curvature perturbation
        power spectrum
    n_s : float
        Cosmology parameter, spectral index of scalar perturbation power
        spectrum

    Returns
    -------
    power_spectrum : (nk, nz) array_like
        Array of values for the linear matter power spectrum in  [Mpc^3]
        evaluated at the input wavenumbers for the given primordial power
        spectrum parameters, cosmology. For nz redshifts and nk wavenumbers
        the returned array will have shape (nk, nz).

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import default_cosmology
    >>> cosmology = default_cosmology.get()
    >>> redshift = np.array([0, 1])
    >>> wavenumber = np.array([1.e-2, 1.e-1, 1e0])
    >>> A_s = 2.e-9
    >>> n_s = 0.965
    >>> classy(wavenumber, redshift, cosmology, A_s, n_s)  # doctest: +SKIP
    array([[2.34758952e+04, 8.70837957e+03],
           [3.03660813e+03, 1.12836115e+03],
           [2.53124880e+01, 9.40802814e+00]])
    References
    ----------
    doi : 10.1088/1475-7516/2011/07/034
    arXiv: 1104.2932, 1104.2933

    """
    try:
        from classy import Class
    except ImportError:
        raise Exception("CLASS is required to use skypy.linear.classy")

    redshift = np.atleast_1d(redshift)

    k = wavenumber * (1. / u.Mpc)
    k_h = k.to((u.littleh / u.Mpc), u.with_H0(cosmology.H0))

    h2 = cosmology.h * cosmology.h

    params = {
        'output': 'mPk',
        'P_k_max_h/Mpc':  k_h.max().value,
        'z_pk': ', '.join(str(z) for z in redshift),
        'A_s':       A_s,
        'n_s':       n_s,
        'H0':        cosmology.H0.value,
        'omega_b':   cosmology.Ob0 * h2,
        'omega_cdm': cosmology.Om0 * h2,
        'T_cmb':     cosmology.Tcmb0.value,
        'N_eff':     cosmology.Neff,
    }

    classy_obj = Class()

    classy_obj.set(params)

    classy_obj.compute()

    pzk = np.zeros([redshift.shape[0], wavenumber.shape[0]])

    for i, ki in enumerate(wavenumber):
        for j, zj in enumerate(redshift):
            pzk[j, i] = classy_obj.pk(ki, zj)

    return pzk.T