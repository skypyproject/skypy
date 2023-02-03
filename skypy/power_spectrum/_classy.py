import numpy as np
from astropy import units
from ._base import TabulatedPowerSpectrum

__all__ = [
    'CLASSY',
]


class CLASSY(TabulatedPowerSpectrum):
    """Return the CLASS computation of the linear matter power spectrum, on a
    two dimensional grid of wavenumber and redshift.
    """

    def __init__(self, kmax, redshift, cosmology, A_s, n_s, tau, **kwargs):
        try:
            from classy import Class
        except ImportError:
            raise Exception("classy is required to use skypy.linear.classy")

        h2 = cosmology.h * cosmology.h

        params = {
            'output': 'mPk',
            'P_k_max_1/Mpc':  kmax,
            'z_pk': ', '.join(str(z) for z in np.atleast_1d(redshift)),
            'A_s':       A_s,
            'n_s':       n_s,
            'H0':        cosmology.H0.value,
            'omega_b':   cosmology.Ob0 * h2,
            'omega_cdm': cosmology.Odm0 * h2,
            'T_cmb':     cosmology.Tcmb0.value,
            'N_ncdm':    cosmology._nmassivenu,
            'N_ur':      cosmology.Neff - cosmology._nmassivenu,
            'tau_reio':  tau,
            'YHe':       0.24
        }

        if cosmology.has_massive_nu:

            params['T_ncdm'] = 0.7133

            if cosmology._nmassivenu == 1:
                params['m_ncdm'] = np.sum(cosmology.m_nu.to_value(units.eV))
            else:
                params['m_ncdm'] = ', '.join(cosmology.m_nu.to_value(units.eV).astype(str))

        params.update(kwargs)

        classy_obj = Class()
        classy_obj.set(params)
        classy_obj.compute()

        p, k, z = classy_obj.get_pk_and_k_and_z(nonlinear=False)
        z_order = np.argsort(z)
        super().__init__(k, z[z_order], p.T[z_order, :])
