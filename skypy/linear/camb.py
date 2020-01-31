import numpy as np
import camb as _camb


def camb(wavenumber, redshift, cosmology):
    """ CAMB computation of the linear matter power spectrum, on a two
    dimensional grid of wavenumber and redshift

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

    h2 = cosmology.h*cosmology.h

    # ToDo: ensure astropy.cosmology can fully specify model
    pars.set_cosmology(H0=cosmology.H0.value,
                        ombh2=cosmology.Ob0*h2,
                        omch2=cosmology.Odm0*h2,
                        omk=cosmology.Ok0,
                        TCMB=cosmology.Tcmb0.value,
                        mnu=np.sum(cosmology.m_nu.value),
                        standard_neutrino_neff=cosmology.Neff
                        #tau=cosmology.tau
                        )

    pars.WantTransfer = True

    # check redshifts decreasing (required by camb)
    if redshifts[-1] > redshifts[0]:
        redshifts = redshifts[::-1]

    #pars.InitPower.ns = cosmology.n
    
    pars.Transfer.PK_num_redshifts = len(list(redshifts))
    pars.Transfer.PK_redshifts = list(redshifts)

    results = _camb.get_results(pars)
    
    kh, z, power_spectrum = results.get_matter_power_spectrum(minkh=wavenumber.min(), maxkh=wavenumber.max(), npoints=len(wavenumber))

    return power_spectrum