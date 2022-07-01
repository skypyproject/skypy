import numpy as np
from astropy import units
from inspect import signature
from ._base import TabulatedPowerSpectrum


__all__ = [
    'CAMB',
    'camb',
]


def camb(wavenumber, redshift, cosmology, A_s, n_s):
    r'''CAMB linear matter power spectrum.
    Compute the linear matter power spectrum on a two dimensional grid of
    redshift and wavenumber using CAMB [1]_.

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
    A_s : float
        Cosmology parameter, amplitude normalisation of curvature perturbation
        power spectrum
    n_s : float
        Cosmology parameter, spectral index of scalar perturbation power
        spectrum

    Returns
    -------
    power_spectrum : (nz, nk) array_like
        Array of values for the linear matter power spectrum in Mpc3
        evaluated at the input wavenumbers for the given primordial power
        spectrum parameters, cosmology. For nz redshifts and nk wavenumbers
        the returned array will have shape (nz, nk).

    References
    ----------
    .. [1] Lewis, A. and Challinor, A. and Lasenby, A. (2000),
        doi : 10.1086/309179.

    '''

    try:
        from camb import CAMBparams, model, get_matter_power_interpolator
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
                       mnu=np.sum(cosmology.m_nu.to_value(units.eV)),
                       standard_neutrino_neff=cosmology.Neff
                       )

    # camb interpolator requires redshifts to be in increasing order
    redshift_order = np.argsort(redshift)
    wavenumber_order = np.argsort(wavenumber)

    pars.InitPower.ns = n_s
    pars.InitPower.As = A_s

    pars.NonLinear = model.NonLinear_none

    if len(redshift) > signature(get_matter_power_interpolator).parameters['nz_step'].default:
        redshift_kwargs = {'zmin': np.min(redshift), 'zmax': np.max(redshift)}
    else:
        redshift_kwargs = {'zs': redshift}

    pk_interp = get_matter_power_interpolator(pars,
                                              nonlinear=False,
                                              hubble_units=False, k_hunit=False,
                                              kmax=np.max(wavenumber),
                                              **redshift_kwargs)

    pzk = pk_interp.P(redshift[redshift_order], wavenumber[wavenumber_order])

    return pzk[redshift_order].reshape(return_shape)


class CAMB(TabulatedPowerSpectrum):

    def __init__(self, kmax, redshift, cosmology, A_s, n_s, **kwargs):

        try:
            from camb import CAMBparams, model, get_results
        except ImportError:
            raise Exception("camb is required to use skypy.power_spectrum.camb")

        h2 = cosmology.h * cosmology.h

        pars = CAMBparams()
        pars.set_cosmology(H0=cosmology.H0.value,
                           ombh2=cosmology.Ob0 * h2,
                           omch2=cosmology.Odm0 * h2,
                           omk=cosmology.Ok0,
                           TCMB=cosmology.Tcmb0.value,
                           mnu=np.sum(cosmology.m_nu.to_value(units.eV)),
                           standard_neutrino_neff=cosmology.Neff
                           )

        pars.InitPower.ns = n_s
        pars.InitPower.As = A_s
        pars.NonLinear = model.NonLinear_none
        k_per_logint, var1, var2, hubble_units, nonlinear = None, None, None, False, False
        pars.set_matter_power(redshifts=redshift, kmax=kmax, k_per_logint=k_per_logint, silent=True)
        results = get_results(pars)
        k, z, p = results.get_linear_matter_power_spectrum(var1, var2, hubble_units,
                                                           nonlinear=nonlinear)

        super().__init__(k*cosmology.h, z, p)
