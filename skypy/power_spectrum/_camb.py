"""CAMB module.

This module facilitates the CAMB computation of the linear
matter power spectrum.
"""

import numpy as np
from astropy import units as u


__all__ = [
    'camb',
]


def camb(wavenumber, redshift, cosmology, A_s, n_s):
    r'''CAMB linear matter power spectrum.
    Return the CAMB computation of the linear matter power spectrum, on a
    two dimensional grid of wavenumber and , described in [1]_.

    Parameters
    ----------
    wavenumber : (nk,) array_like
        Array of wavenumbers in units of :math:`[Mpc^-1]` at which to
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
        Array of values for the linear matter power spectrum in :math:`[Mpc^3]`
        evaluated at the input wavenumbers for the given primordial power
        spectrum parameters, cosmology. For nz redshifts and nk wavenumbers
        the returned array will have shape (nz, nk).

    Examples
    --------
    >>> import numpy as np
    >>> from skypy.power_spectrum import camb
    >>> from astropy.cosmology import default_cosmology, Planck15
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP

    This will return the linear matter power spectrum in :math:`Mpc^3`
    at several values of redshift and wavenumers in :math:`1/Mpc`
    for the Astropy default cosmology:

    >>> cosmology = default_cosmology.get()
    >>> redshift = np.array([0, 1])
    >>> wavenumber = np.array([1.e-2, 1.e-1, 1e0])
    >>> A_s = 2.e-9
    >>> n_s = 0.965
    >>> camb(wavenumber, redshift, cosmology, A_s, n_s)  # doctest: +SKIP
    array([[2.36646871e+04, 3.02592011e+03, 2.49336836e+01],
       [8.77864738e+03, 1.12441960e+03, 9.26749240e+00]])

    Create a plot for the linear matter power spectrum at redshift 0
    and Planck15 cosmology:

    >>> redshift = 0.0
    >>> wavenumber = np.logspace(-4.0, 0.0, 100)
    >>> A_s, ns = 2.e-9, 0.965
    >>> pc = camb(wavenumber, redshift, cosmology, A_s, n_s)
    >>> plt.loglog(k, smith.T, label='Smith')  # doctest: +SKIP
    >>> plt.loglog(k,p.T, label='Linear')  # doctest: +SKIP
    >>> plt.xlabel(r'k $(1/Mpc)$')  # doctest: +SKIP
    >>> plt.ylabel(r'P $(Mpc^3)$')  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

    References
    ----------
    .. [1] Lewis, A. and Challinor, A. and Lasenby, A. (2000),
        doi : 10.1086/309179.

    '''

    try:
        from camb import CAMBparams, get_results, model
    except ImportError:
        raise Exception("camb is required to use skypy.linear.camb")

    return_shape = np.shape(wavenumber)
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

    if len(z) > 1:
        power_output = pzk[redshift_order[::-1]]
    elif len(z) == 1:
        power_output = pzk[redshift_order[::-1]].reshape(return_shape)

    return power_output
