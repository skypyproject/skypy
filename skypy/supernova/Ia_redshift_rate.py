r"""Creates the volumetric SNe Ia rate for an array of redshifts
=================
.. autosummary::
   :nosignatures:
   :toctree: ../api/
   SNeIa_redshifts

Models
======
.. autosummary::
   :nosignatures:
   :toctree: ../api/
   Ia_redshift_rate
   
"""

from astropy import units

def Ia_redshift_rate(redshift,r0 = 2.27e-5,a = 1.7):
    """Creates a redshift distribution of Type Ia Supernovae
    This function computes the redshift distribution of Type Ia Supernovae
    using the rate parameters as given in [1].
    Parameters
    ----------
    redshift : numpy.array
        Redshift at which to evaluate the Type Ia supernovae rate.
    Returns
    -------
    rate : astropy.Quantity
        Volumetric rate of Type Ia's the redshifts given in units of [SNe Ia yr−1 Mpc−3]

    Examples
    --------
    >>> import numpy as np
    >>> z = np.array([0.0,0.1,0.2])
    >>> Ia_redshift_rate(z,r0=2.27e-5,a=1.7)
    <Quantity [2.27000000e-05, 2.66927563e-05, 3.09480989e-05] yr / Mpc3>
    References
    ----------
    .. [1] Frohmaier, C. et al. (2019), https://arxiv.org/pdf/1903.08580.pdf .
    """

    rate = r0*(1+redshift)**a * units.year *(units.Mpc)**-3
    return rate