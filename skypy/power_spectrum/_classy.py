import numpy as np

__all__ = [
    'classy',
]


def classy(wavenumber, redshift, cosmology, **kwargs):
    """ Return the CLASS computation of the linear matter power spectrum, on a
    two dimensional grid of wavenumber and redshift.

    Additional CLASS parameters can be passed via keyword arguments.

    Parameters
    ----------
    wavenumber : (nk,) array_like
        Array of wavenumbers in units of Mpc-1 at which to
        evaluate the linear matter power spectrum.
    redshift : (nz,) array_like
        Array of redshifts at which to evaluate the linear matter power
        spectrum.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing omega_matter, omega_baryon, Hubble
        parameter and CMB temperature in the present day

    Returns
    -------
    power_spectrum : (nz, nk) array_like
        Array of values for the linear matter power spectrum in Mpc3
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
    >>> z_reio = 10.
    >>> classy(wavenumber, redshift, cosmology, A_s, n_s, z_reio)  # doctest: +SKIP
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
        raise Exception("classy is required to use skypy.linear.classy")

    h2 = cosmology.h * cosmology.h

    params = {
        'output': 'mPk',
        'P_k_max_1/Mpc':  np.max(wavenumber),
        'z_pk': ', '.join(str(z) for z in np.atleast_1d(redshift)),
        'H0':        cosmology.H0.value,
        'omega_b':   cosmology.Ob0 * h2,
        'omega_cdm': cosmology.Odm0 * h2,
        'T_cmb':     cosmology.Tcmb0.value,
        'N_eff':     cosmology.Neff,
    }

    params.update(kwargs)

    classy_obj = Class()
    classy_obj.set(params)
    classy_obj.compute()

    z = np.expand_dims(redshift, (-1,)*np.ndim(wavenumber))
    k = np.expand_dims(wavenumber, (0,)*np.ndim(redshift))
    z, k = np.broadcast_arrays(z, k)
    pzk = np.empty(z.shape)

    for i in np.ndindex(*pzk.shape):
        pzk[i] = classy_obj.pk_lin(k[i], z[i])

    if pzk.ndim == 0:
        pzk = pzk.item()

    return pzk
