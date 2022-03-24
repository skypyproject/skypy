r"""Galaxy spectrum module.

"""

from astropy import units
from astropy.io import fits
import numpy as np
from pkg_resources import resource_filename
from skypy.utils.photometry import SpectrumTemplates


__all__ = [
    'dirichlet_coefficients',
    'KCorrectTemplates',
    'kcorrect',
]


def dirichlet_coefficients(redshift, alpha0, alpha1, z1=1., weight=None):
    r"""Dirichlet-distributed SED coefficients.

    Spectral coefficients to calculate the rest-frame spectral energy
        distribution of a galaxy following the Herbel et al. model in [1]_.

    Parameters
    ----------
    redshift : (nz,) array_like
        The redshift values of the galaxies for which the coefficients want to
        be sampled.
    alpha0, alpha1 : (nc,) array_like
        Factors parameterising the Dirichlet distribution according to Equation
        (3.9) in [1]_.
    z1 : float or scalar, optional
       Reference redshift at which alpha = alpha1. The default value is 1.0.
    weight : (nc,) array_like, optional
        Different weights for each component.

    Returns
    -------
    coefficients : (nz, nc) ndarray
        The spectral coefficients of the galaxies. The shape is (n, nc) with nz
        the number of redshifts and nc the number of coefficients.

    Notes
    -----
    One example is the rest-frame spectral energy distribution of galaxies
    which can be written as a linear combination of the five kcorrect [2]_
    template spectra :math:`f_i`
    (see [1]_)

    .. math::

        f(\lambda) = \sum_{i=1}^5 c_i f_i(\lambda) \;,

    where the coefficients :math:`c_i` were shown to follow a Dirichlet
    distribution of order 5. The five parameters describing the Dirichlet
    distribution are given by

    .. math::

        \alpha_i(z) = (\alpha_{i,0})^{1-z/z_1} \cdot (\alpha_{i,1})^{z/z_1} \;.

    Here, :math:`\alpha_{i,0}` describes the galaxy population at redshift
    :math:`z=0` and :math:`\alpha_{i,1}` the population at :math:`z=z_1 > 0`.
    These parameters depend on the galaxy type and we chose :math:`z_1=1`.

    Beside this example, this code works for a general number of templates.

    References
    ----------
    .. [1] Herbel J., Kacprzak T., Amara A. et al., 2017, Journal of Cosmology
           and Astroparticle Physics, Issue 08, article id. 035 (2017)
    .. [2] Blanton M. R., Roweis S., 2007, The Astronomical Journal,
           Volume 133, Page 734

    """

    if np.ndim(alpha0) != 1 or np.ndim(alpha1) != 1:
        raise ValueError('alpha0, alpha1 must be 1D arrays')
    if len(alpha0) != len(alpha1):
        raise ValueError('alpha0 and alpha1 must have the same length')
    if weight is not None and (np.ndim(weight) != 1 or len(weight) != len(alpha0)):
        raise ValueError('weight must be 1D and match alpha0, alpha1')

    redshift = np.expand_dims(redshift, -1)

    alpha = np.power(alpha0, 1-redshift/z1)*np.power(alpha1, redshift/z1)

    # sample Dirichlet by normalising independent gamma draws
    coeff = np.random.gamma(alpha)
    if weight is not None:
        coeff *= weight
    coeff /= coeff.sum(axis=-1)[..., np.newaxis]

    return coeff


class KCorrectTemplates(SpectrumTemplates):
    '''Galaxy spectra from kcorrect templates.

    Class for modeling galaxy spectra as a linear combination of the five
    kcorrect template spectra [1]_.

    References
    ----------
    .. [1] M. R. Blanton and S. Roweis, 2007, AJ, 125, 2348
    '''

    def __init__(self, hdu=1):
        filename = resource_filename('skypy', 'data/kcorrect/k_nmf_derived.default.fits')
        with fits.open(filename) as hdul:
            self.templates = hdul[hdu].data * units.Unit('erg s-1 cm-2 angstrom-1')
            self.wavelength = hdul[11].data * units.Unit('angstrom')
            self.mass = hdul[16].data
            self.mremain = hdul[17].data
            self.mets = hdul[18].data
            self.mass300 = hdul[19].data
            self.mass1000 = hdul[20].data

    def stellar_mass(self, coefficients, magnitudes, filter):
        r'''Compute stellar mass from absolute magnitudes in a reference filter.

        This function takes composite spectra for a set of galaxies defined by
        template fluxes *per unit stellar mass* and multiplicative coefficients
        and calculates the stellar mass required to match given absolute
        magnitudes for a given bandpass filter in the rest frame.

        Parameters
        ----------
        coefficients : (ng, nt) array_like
            Array of template coefficients.
        magnitudes : (ng,) array_like
            The magnitudes to match in the reference bandpass.
        filter : str
            A single reference bandpass filter specification for
            `~speclite.filters.load_filters`.

        Returns
        -------
        stellar_mass : (ng,) array_like
            Stellar mass of each galaxy in template units.
        '''
        Mt = self.absolute_magnitudes(coefficients, filter)
        return np.power(10, 0.4*(Mt-magnitudes))

    def metallicity(self, coefficients):
        r'''Galaxy metallicities from kcorrect templates.

        This function calculates the matallicities of galaxies modelled as a
        linear combination of the kcorrect templates [1]_.

        Parameters
        ----------
        coefficients : (ng, 5) array_like
            Array of template coefficients.

        Returns
        -------
        metallicity : (ng,) array_like
            Metallicity of each galaxy.

        References
        ----------
        .. [1] M. R. Blanton and S. Roweis, 2007, AJ, 125, 2348

        '''

        return np.sum(coefficients * self.mremain * self.mets) / np.sum(coefficients * self.mremain)

    def m300(self, coefficients, stellar_mass=None):
        r'''Stellar mass formed in the last 300 Myr.

        This function calculates the mass of new stars formed within the last
        300 Myr for galaxies modelled as a linear combination of the kcorrect
        templates [1]_.

        Parameters
        ----------
        coefficients : (ng, 5) array_like
            Array of template coefficients.
        stellar_mass : (ng,) array_like, optional
            Optional array of stellar masses for each galaxy.

        Returns
        -------
        m300 : (ng,) array_like
            Total mass of new stars formed in the last 300 Myr as a fraction of
            the stellar mass of each galaxy. If stellar_mass is given, instead
            returns the absolute mass of new stars.

        References
        ----------
        .. [1] M. R. Blanton and S. Roweis, 2007, AJ, 125, 2348

        '''

        sm = stellar_mass if stellar_mass is not None else 1
        return sm * np.sum(coefficients * self.mass300) / np.sum(coefficients * self.mass)

    def m1000(self, coefficients, stellar_mass=None):
        r'''Stellar mass formed in the last 1 Gyr.

        This function calculates the mass of new stars formed within the last
        1 Gyr for galaxies modelled as a linear combination of the kcorrect
        templates [1]_.

        Parameters
        ----------
        coefficients : (ng, 5) array_like
            Array of template coefficients.
        stellar_mass : (ng,) array_like, optional
            Optional array of stellar masses for each galaxy.

        Returns
        -------
        m1000 : (ng,) array_like
            Total mass of new stars formed in the last 1 Gyr as a fraction of
            the stellar mass of each galaxy. If stellar_mass is given, instead
            returns the absolute mass of new stars.

        References
        ----------
        .. [1] M. R. Blanton and S. Roweis, 2007, AJ, 125, 2348

        '''

        sm = stellar_mass if stellar_mass is not None else 1
        return sm * np.sum(coefficients * self.mass1000) / np.sum(coefficients * self.mass)


kcorrect = KCorrectTemplates(hdu=1)
'''`KCorrectTemplates` using kcorrect smoothed templates.'''
