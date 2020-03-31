import numpy as np
from camb import CAMBparams, get_results, model
from astropy import units as uns


def camb(wavenumber, redshift, cosmology, A_s, n_s):
    """ Return the CAMB computation of the linear matter power spectrum, on a
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
        Cosmology parameter, spectral infex of scalar perturbation power
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
    >>> wavenumber = np.array([1.e-2, 1.e-1])
    >>> A_s = 2.e-9
    >>> n_s = 0.965
    >>> camb(wavenumber, redshift, cosmology, A_s, n_s)
    array([[17596.19571205,  9367.99583637],
           [ 6524.28734592,  3479.62135542]])

    Reference
    ---------
    doi : 10.1086/309179
    arXiv: astro-ph/9911177

    """

    redshift = np.atleast_1d(redshift)

    h2 = cosmology.h * cosmology.h

    # ToDo: ensure astropy.cosmology can fully specify model
    pars = CAMBparams()
    pars.set_cosmology(H0=cosmology.H0.value,
                       ombh2=cosmology.Ob0 * h2,
                       omch2=cosmology.Odm0 * h2,
                       omk=cosmology.Ok0,
                       TCMB=cosmology.Tcmb0.value,
                       mnu=np.sum(cosmology.m_nu.value),
                       standard_neutrino_neff=cosmology.Neff
                       )

    redshift_order = np.argsort(redshift)[::-1]  # camb requires redshifts to
                                                 # be in decreasing order

    pars.InitPower.ns = n_s
    pars.InitPower.As = A_s

    pars.set_matter_power(redshifts=list(redshift[redshift_order]),
                          kmax=np.max(wavenumber))

    pars.NonLinear = model.NonLinear_none

    results = get_results(pars)

    k = wavenumber * (1. / uns.Mpc)

    k_h = k.to((uns.littleh / uns.Mpc), uns.with_H0(cosmology.H0))

    kh, z, power_spectrum = results.get_matter_power_spectrum(minkh=np.min(k_h.value),
                                                              maxkh=np.max(k_h.value),
                                                              npoints=len(k_h.value))

    return power_spectrum[redshift_order[::-1]].T
