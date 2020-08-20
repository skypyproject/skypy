import numpy as np
from astropy import units as u


__all__ = [
    'camb',
]


def camb(wavenumber, redshift, cosmology, A_s, n_s):
    r'''CAMB linear matter power spectrum.
    Compute the linear matter power spectrum on a two dimensional grid of
    redshift and wavenumber using CAMB [1]_.

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
    power_spectrum : (nz, nk) array_like
        Array of values for the linear matter power spectrum in  [Mpc^3]
        evaluated at the input wavenumbers for the given primordial power
        spectrum parameters, cosmology. For nz redshifts and nk wavenumbers
        the returned array will have shape (nz, nk).

    Examples
    --------

    >>> import numpy as np
    >>> from astropy.cosmology import default_cosmology
    >>> cosmology = default_cosmology.get()
    >>> redshift = np.array([0, 1])
    >>> wavenumber = np.array([1.e-2, 1.e-1, 1e0])
    >>> A_s = 2.e-9
    >>> n_s = 0.965
    >>> power_spectrum = camb(wavenumber, redshift, cosmology, A_s, n_s)

    References
    ----------
    .. [1] Lewis, A. and Challinor, A. and Lasenby, A. (2000),
        doi : 10.1086/309179.

    '''

    try:
        from camb import CAMBparams, get_results, model
    except ImportError:
        raise Exception("camb is required to use skypy.power_spectrum.camb")

    return_shape = (*np.shape(redshift), *np.shape(wavenumber))
    redshift = np.atleast_1d(redshift)

    h2 = cosmology.h * cosmology.h

    pars = CAMBparams()
    pars.set_cosmology(H0=cosmology.H0.value,
                       ombh2=cosmology.Ob0 * h2,
                       omch2=cosmology.Odm0 * h2,
                       omk=cosmology.Ok0,
                       TCMB=cosmology.Tcmb0.value,
                       mnu=np.sum(cosmology.m_nu.value),
                       standard_neutrino_neff=cosmology.Neff
                       )

    # camb requires redshifts to be in decreasing order
    redshift_order = np.argsort(redshift)[::-1]

    pars.InitPower.ns = n_s
    pars.InitPower.As = A_s

    pars.set_matter_power(redshifts=list(redshift[redshift_order]),
                          kmax=np.max(wavenumber))

    pars.NonLinear = model.NonLinear_none

    results = get_results(pars)

    k = wavenumber * (1. / u.Mpc)

    k_h = k.to((u.littleh / u.Mpc), u.with_H0(cosmology.H0))

    kh, z, pzk = results.get_matter_power_spectrum(minkh=np.min(k_h.value),
                                                   maxkh=np.max(k_h.value),
                                                   npoints=len(k_h.value))

    return pzk[redshift_order[::-1]].reshape(return_shape)
