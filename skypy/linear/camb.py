import numpy as np
import camb as _camb


def camb(wavenumber, redshift, cosmology):
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
    Returns
    -------
    power_spectrum : array_like
        Array of values for the linear matter power spectrum evaluated at the
        input wavenumbers for the given primordial power spectrum parameters,
        cosmology
    References
    ----------

    """

    print('Using CAMB %s installed at %s'%(_camb.__version__,os.path.dirname(_camb.__file__)))

    redshifts = np.atleast_1d(redshifts)

    pars.WantTransfers = True
    pars.Transfer.PK_num_redshifts = len(list(redshifts))
    pars.Transfer.PK_redshifts = list(redshifts)

    results = _camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=wavenumber.min(), maxkh=wavenumber.max(), npoints=len(wavenumber))
    return power_spectrum