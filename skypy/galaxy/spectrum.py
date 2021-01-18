r"""Galaxy spectrum module.

"""

import numpy as np
from astropy import units
from astropy.io import fits
from pkg_resources import resource_filename
from ..utils import spectral_data_input


__all__ = [
    'dirichlet_coefficients',
    'load_spectral_data',
    'ab_maggies_redshift',
    'template_absolute_magnitudes',
    'template_apparent_magnitudes',
    'kcorrect_absolute_magnitudes',
    'kcorrect_apparent_magnitudes',
    'kcorrect_stellar_mass',
]

try:
    __import__('specutils')
except ImportError:
    HAS_SPECUTILS = False
else:
    HAS_SPECUTILS = True

try:
    import speclite.filters
except ImportError:
    HAS_SPECLITE = False
else:
    HAS_SPECLITE = True


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

    Examples
    --------
    >>> from skypy.galaxy.spectrum import dirichlet_coefficients
    >>> import numpy as np

    Sample the coefficients according to [1]_ for n blue galaxies with
    redshifts between 0 and 1.

    >>> n = 100000
    >>> alpha0 = np.array([2.079, 3.524, 1.917, 1.992, 2.536])
    >>> alpha1 = np.array([2.265, 3.862, 1.921, 1.685, 2.480])
    >>> redshift = np.linspace(0,2, n)
    >>> coefficients = dirichlet_coefficients(redshift, alpha0, alpha1)

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


def ab_maggies_redshift(wavelength, spectrum, filters, redshift, interpolate=1000):
    r'''Compute AB fluxes from spectra and filters.

    This function takes *emission* spectra and observation filters and computes
    bandpass AB fluxes in units of maggies [1]_.

    The filter specification in the `filters` argument is passed unchanged to
    `speclite.filters.load_filters`. See there for the syntax, and the list of
    supported values.

    The emission spectra can optionally be redshifted. If the `redshift`
    parameter is given, the output array will have corresponding axes. If the
    `interpolate` parameter is not `False`, at most that number of redshifted
    spectra are computed, and the remainder is interpolated from the results.

    Parameters
    ----------
    wavelength : (nw,) `~astropy.units.Quantity` or array_like
        Wavelength array of the spectrum.
    spectrum : ([ns,] nw,) `~astropy.units.Quantity` or array_like
        Emission spectrum. Can be multidimensional for computing with multiple
        spectra of the same wavelengths. The last axis is the wavelength axis.
    filters : str or list of str
        Filter specification, loaded filters are array_like of shape (nf,).
    redshift : (nz,) array_like
        Array of redshifts. Can be multidimensional.
    interpolate : int or `False`, optional
        Maximum number of redshifts to compute explicitly. Default is `1000`.

    Returns
    -------
    flux : ([nz,] [ns,] nf,) array_like
        The AB flux of each redshift (if given), each spectrum, and each filter.

    Warnings
    --------
    The :mod:`speclite` package must be installed to use this function.

    References
    ----------
    .. [1] M. R. Blanton et al., 2003, AJ, 125, 2348

    '''

    # load the filters
    if np.ndim(filters) == 0:
        filters = (filters,)
    filters = speclite.filters.load_filters(*filters)
    if np.shape(filters) == (1,):
        filters = filters[0]

    # number of dimensions for each input
    nd_s = len(np.shape(spectrum)[:-1])  # last axis is spectral axis
    nd_f = len(np.shape(filters))
    nd_z = len(np.shape(redshift))

    # check if interpolation is necessary
    if interpolate and np.size(redshift) <= interpolate:
        interpolate = False

    # if interpolating, split the redshift range into `interpolate` bits
    if interpolate:
        redshift_ = np.linspace(np.min(redshift), np.max(redshift), interpolate)
    else:
        redshift_ = redshift if redshift is not None else 0

    # working array shape
    m_shape = np.shape(redshift_) + np.shape(spectrum)[:-1] + np.shape(filters)

    # compute AB maggies for every redshift, spectrum, and filter
    m = np.empty(m_shape)
    for i, z in np.ndenumerate(redshift_):
        for j, f in np.ndenumerate(filters):
            # create a shifted filter for redshift
            fs = f.create_shifted(z)
            m[i+(...,)+j] = fs.get_ab_maggies(spectrum, wavelength)

    # if interpolating, compute the full set of redshifts
    if interpolate:
        # diy interpolation keeps memory use to a minimum
        dm = np.diff(m, axis=0, append=m[-1:])
        u, n = np.modf(np.interp(redshift, redshift_, np.arange(redshift_.size)))
        n = n.astype(int)
        u = u.reshape(u.shape + (1,)*(nd_s+nd_f))
        m = np.ascontiguousarray(m[n])
        m += u*dm[n]
        del(dm, n, u)

    # K-correction
    return m * np.reshape((1 + redshift), np.shape(redshift) + (1,)*(nd_s+nd_f))


def template_absolute_magnitudes(wavelength, templates, coefficients, filters, stellar_mass=None):
    '''Galaxy AB absolute magnitudes from template spectra.

    This function calculates photometric AB absolute magnitudes for galaxies
    whose spectra are modelled as a linear combination of a set of template
    spectra.

    Parameters
    ----------
    wavelength : (nw,) `~astropy.units.Quantity` or array_like
        Wavelength array of the spectra.
    templates : (nt, nw,) `~astropy.units.Quantity` or array_like
        Emission spectra of the template set.
    coefficients : (ng, nt) array_like
        Array of spectrum coefficients.
    filters : (nf,) str or list of str
        Bandpass filter specification for `~speclite.filters.load_filters`.
    stellar_mass : (ng,) array_like, optional
        Optional array of stellar masses for each galaxy in template units.

    Returns
    -------
    magnitudes : (ng, nf) array_like
        The absolute AB magnitude of each object in each filter, where ``nf``
        is the number of loaded filters.
    '''

    flux = ab_maggies_redshift(wavelength, templates, filters, 0)
    flux = np.sum((coefficients.T * flux.T).T, axis=1)
    mass_modulus = 0 if stellar_mass is None else -2.5 * np.log10(stellar_mass)

    return (-2.5*np.log10(flux).T + mass_modulus).T


def template_apparent_magnitudes(wavelength, templates, coefficients, redshift, filters,
                                 cosmology, *, stellar_mass=None, resolution=1000):
    '''Galaxy AB apparent magnitudes from template spectra.

    This function calculates photometric AB apparent magnitudes for galaxies
    whose spectra are modelled as a linear combination of a set of template
    spectra.

    Parameters
    ----------
    wavelength : (nw,) `~astropy.units.Quantity` or array_like
        Wavelength array of the spectra.
    templates : (nt, nw,) `~astropy.units.Quantity` or array_like
        Emission spectra of the template set.
    coefficients : (ng, nt) array_like
        Array of spectrum coefficients.
    redshifts : (ng,) array_like
        Array of redshifts for each galaxy used to calculte the distance
        modulus and k-correction.
    filters : (nf,) str or list of str
        Bandpass filter specification for `~speclite.filters.load_filters`.
    cosmology : Cosmology
        Astropy Cosmology object to calculate distance modulus.
    stellar_mass : (ng,) array_like, optional
        Optional array of stellar masses for each galaxy in template units.
    resolution : integer, optional
        Redshift resolution for intepolating magnitudes. Default is 1000. If
        the number of objects is less than resolution their magnitudes are
        calculated directly without interpolation.

    Returns
    -------
    magnitudes : (ng, nf) array_like
        The apparent AB magnitude of each object in each filter.
    '''

    flux = ab_maggies_redshift(wavelength, templates, filters, redshift, resolution)
    flux = np.sum((coefficients.T * flux.T).T, axis=1)
    distance_modulus = cosmology.distmod(redshift).value
    mass_modulus = 0 if stellar_mass is None else -2.5 * np.log10(stellar_mass)

    return (-2.5*np.log10(flux).T + distance_modulus + mass_modulus).T


def kcorrect_absolute_magnitudes(coefficients, filters, stellar_mass=None):
    '''Galaxy AB absolute magnitudes from kcorrect template spectra.

    This function calculates photometric AB absolute magnitudes for galaxies
    whose spectra are modelled as a linear combination of the kcorrect template
    spectra from [1]_ and [2]_.

    Parameters
    ----------
    coefficients : (ng, nt) array_like
        Array of spectrum coefficients.
    filters : str or list of str
        Bandpass filter specification for `~speclite.filters.load_filters`.
    stellar_mass : (ng,) array_like, optional
        Optional array of stellar masses for each galaxy in template units.

    Returns
    -------
    mag_ab : (ng, nf) array_like
        The absolute AB magnitude of each object in each filter, where ``nf``
        is the number of loaded filters.

    References
    ----------
    .. [1] M. R. Blanton et al., 2003, AJ, 125, 2348
    .. [2] M. R. Blanton and S. Roweis, 2007, AJ, 125, 2348
    '''

    # kcorrect data
    filename = resource_filename('skypy', 'data/kcorrect/k_nmf_derived.default.fits')
    with fits.open(filename) as hdul:
        spec = hdul[1].data * units.Unit('erg s-1 cm-2 angstrom-1')
        lambda_ = hdul[11].data * units.Unit('angstrom')

    return template_absolute_magnitudes(lambda_, spec, coefficients, filters, stellar_mass)


def kcorrect_apparent_magnitudes(coefficients, redshift, filters, cosmology,
                                 *, stellar_mass=None, resolution=1000):
    '''Galaxy AB apparent magnitudes from kcorrect template spectra.

    This function calculates photometric AB apparent magnitudes for galaxies
    whose spectra are modelled as a linear combination of the kcorrect template
    spectra from [1]_ and [2]_.

    Parameters
    ----------
    coefficients : (ng, nt) array_like
        Array of spectrum coefficients.
    redshifts : (ng,) array_like
        Array of redshifts for each galaxy used to calculte the distance
        modulus and k-correction.
    filters : str or list of str
        Bandpass filter specification for `~speclite.filters.load_filters`.
    cosmology : Cosmology
        Astropy Cosmology object to calculate distance modulus.
    stellar_mass : (ng,) array_like, optional
        Optional array of stellar masses for each galaxy in template units.
    resolution : integer, optional
        Redshift resolution for intepolating magnitudes. Default is 1000. If
        the number of objects is less than resolution their magnitudes are
        calculated directly without interpolation.

    Returns
    -------
    mag_ab : (ng, nf) array_like
        The absolute AB magnitude of each object in each filter, where ``nf``
        is the number of loaded filters.

    References
    ----------
    .. [1] M. R. Blanton et al., 2003, AJ, 125, 2348
    .. [2] M. R. Blanton and S. Roweis, 2007, AJ, 125, 2348
    '''

    # kcorrect data
    filename = resource_filename('skypy', 'data/kcorrect/k_nmf_derived.default.fits')
    with fits.open(filename) as hdul:
        spec = hdul[1].data * units.Unit('erg s-1 cm-2 angstrom-1')
        lambda_ = hdul[11].data * units.Unit('angstrom')

    return template_apparent_magnitudes(lambda_, spec, coefficients, redshift, filters,
                                        cosmology, stellar_mass, resolution)


def kcorrect_stellar_mass(coefficients, magnitudes, filter):
    r'''Compute stellar mass from absolute magnitudes in a reference filter.

    This function takes composite spectra for a set of galaxies defined by
    template fluxes *per unit stellar mass* and multiplicative coefficients and
    calculates the stellar mass required to match given absolute magnitudes for
    a given bandpass filter in the rest frame.

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

    # kcorrect data
    filename = resource_filename('skypy', 'data/kcorrect/k_nmf_derived.default.fits')
    with fits.open(filename) as hdul:
        spec = hdul[1].data * units.Unit('erg s-1 cm-2 angstrom-1')
        lambda_ = hdul[11].data * units.Unit('angstrom')

    fluxes = speclite.filters.load_filter(filter).get_ab_maggies(spec, lambda_)
    return 10**(-0.4*magnitudes) / np.sum(coefficients * fluxes, axis=-1)


def load_spectral_data(name):
    '''Load spectral data from a known source or a local file.

    If the given name refers to a known source, the associated spectral data is
    constructed by its designated loader. If no source with the given name is
    found, it is assumed to be a filename.

    Parameters
    ----------
    name : str or list of str
        The name of the spectral data to load, or a list of multiple names.

    Returns
    -------
    spectrum : `~specutils.Spectrum1D` or `~specutils.SpectrumList`
        The spectral data. The wavelength or frequency column is the
        `~specutils.Spectrum1D.spectral_axis` (with units) of the returned
        spectrum, and the spectral column or columns are the
        `~specutils.Spectrum1D.flux` array (with units) of the returned
        spectrum.

    Warnings
    --------
    The :mod:`specutils` package must be installed to use this function.
    The :mod:`speclite` package must be installed to use this function.

    '''

    import re

    # loaders registry
    from ._spectrum_loaders import spectrum_loaders, combine_spectra

    # check non-string input
    if not isinstance(name, str):
        # recurse on lists
        if hasattr(name, '__iter__'):
            spectra = None
            for name_ in name:
                spectra = combine_spectra(spectra, load_spectral_data(name_))
            return spectra
        else:
            raise TypeError('name: not a string or list of strings')

    # go through loaders
    for pattern, loader, *args in spectrum_loaders:
        # try to match given name against pattern
        match = re.fullmatch(pattern, name)
        if match:
            # collect nonempty group matches
            groups = [g for g in match.groups() if g]
            break

    # run the loader
    return loader(*args, *groups)
