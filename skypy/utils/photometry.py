"""Photometry module.

"""


from abc import ABCMeta, abstractmethod
import numpy as np


__all__ = [
    'absolute_magnitude_from_luminosity',
    'luminosity_from_absolute_magnitude',
    'luminosity_in_band',
    'mag_ab',
    'SpectrumTemplates',
]

try:
    import speclite.filters  # noqa F401
except ImportError:
    HAS_SPECLITE = False
else:
    HAS_SPECLITE = True


def mag_ab(wavelength, spectrum, filters, *, redshift=None, coefficients=None,
           distmod=None, interpolate=1000):
    r'''Compute AB magnitudes from spectra and filters.

    This function takes *emission* spectra and observation filters and computes
    bandpass AB magnitudes [1]_.

    The filter specification in the `filters` argument is passed unchanged to
    `speclite.filters.load_filters`. See there for the syntax, and the list of
    supported values.

    The emission spectra can optionally be redshifted. If the `redshift`
    parameter is given, the output array will have corresponding axes. If the
    `interpolate` parameter is not `False`, at most that number of redshifted
    spectra are computed, and the remainder is interpolated from the results.

    The spectra can optionally be combined. If the `coefficients` parameter is
    given, its shape must match `spectra`, and the corresponding axes are
    contracted using a product sum. If the spectra are redshifted, the
    coefficients array can contain axes for each redshift.

    By default, absolute magnitudes are returned. To compute apparent magnitudes
    instead, provide the `distmod` argument with the distance modulus for each
    redshift. The distance modulus is applied after redshifts and coefficients
    and should match the shape of the `redshift` array.

    Parameters
    ----------
    wavelength : (nw,) `~astropy.units.Quantity` or array_like
        Wavelength array of the spectrum.
    spectrum : ([ns,] nw,) `~astropy.units.Quantity` or array_like
        Emission spectrum. Can be multidimensional for computing with multiple
        spectra of the same wavelengths. The last axis is the wavelength axis.
    filters : str or list of str
        Filter specification, loaded filters are array_like of shape (nf,).
    redshift : (nz,) array_like, optional
        Optional array of redshifts. Can be multidimensional.
    coefficients : ([nz,] [ns,]) array_like
        Optional coefficients for combining spectra. Axes must be compatible
        with all redshift and spectrum dimensions.
    distmod : (nz,) array_like, optional
        Optional distance modulus for each redshift. Shape must be compatible
        with redshift dimensions.
    interpolate : int or `False`, optional
        Maximum number of redshifts to compute explicitly. Default is `1000`.

    Returns
    -------
    mag_ab : ([nz,] [ns,] nf,) array_like
        The AB magnitude of each redshift (if given), each spectrum (if not
        combined), and each filter.

    Warnings
    --------
    The :mod:`speclite` package must be installed to use this function.

    References
    ----------
    .. [1] M. R. Blanton et al., 2003, AJ, 125, 2348

    '''
    from speclite.filters import load_filters

    # load the filters
    if np.ndim(filters) == 0:
        filters = (filters,)
    filters = load_filters(*filters)
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

    # combine spectra if asked to
    if coefficients is not None:
        # contraction over spectrum axes (`nd_z` to `nd_z+nd_s`)
        if nd_z == 0:
            m = np.matmul(coefficients, m)
        else:
            c = np.reshape(coefficients, np.shape(coefficients) + (1,)*nd_f)
            m = np.sum(m*c, axis=tuple(range(nd_z, nd_z+nd_s)))
        # no spectrum axes left
        nd_s = 0

    # convert maggies to magnitudes
    np.log10(m, out=m)
    m *= -2.5

    # apply the redshift K-correction if necessary
    if redshift is not None:
        kcorr = -2.5*np.log10(1 + redshift)
        m += np.reshape(kcorr, kcorr.shape + (1,)*(nd_s+nd_f))

    # add distance modulus if given
    if distmod is not None:
        m += np.reshape(distmod, np.shape(distmod) + (1,)*(nd_s+nd_f))

    # done
    return m


class SpectrumTemplates(metaclass=ABCMeta):
    '''Base class for composite galaxy spectra from a set of basis templates'''

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    def absolute_magnitudes(self, coefficients, filters, stellar_mass=None):
        '''Galaxy AB absolute magnitudes from template spectra.

        This function calculates photometric AB absolute magnitudes for
        galaxies whose spectra are modelled as a linear combination of a set of
        template spectra.

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
        magnitudes : (ng, nf) array_like
            The absolute AB magnitude of each object in each filter, where
            ``nf`` is the number of loaded filters.
        '''
        mass_modulus = -2.5*np.log10(stellar_mass) if stellar_mass is not None else 0
        M = mag_ab(self.wavelength, self.templates, filters, coefficients=coefficients)
        return (M.T + mass_modulus).T

    def apparent_magnitudes(self, coefficients, redshift, filters, cosmology, *,
                            stellar_mass=None, resolution=1000):
        '''Galaxy AB apparent magnitudes from template spectra.

        This function calculates photometric AB apparent magnitudes for
        galaxies whose spectra are modelled as a linear combination of a set of
        template spectra.

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
        magnitudes : (ng, nf) array_like
            The apparent AB magnitude of each object in each filter, where
            ``nf`` is the number of loaded filters.
        '''
        distmod = cosmology.distmod(redshift).value
        mass_modulus = -2.5*np.log10(stellar_mass) if stellar_mass is not None else 0
        m = mag_ab(self.wavelength, self.templates, filters, redshift=redshift,
                   coefficients=coefficients, distmod=distmod, interpolate=resolution)
        return (m.T + mass_modulus).T


luminosity_in_band = {
    'Lsun_U': 6.33,
    'Lsun_B': 5.31,
    'Lsun_V': 4.80,
    'Lsun_R': 4.60,
    'Lsun_I': 4.51,
}
'''Bandpass magnitude of reference luminosities.

These values can be used for conversion in `absolute_magnitude_from_luminosity`
and `luminosity_from_absolute_magnitude`. The `Lsun_{UBVRI}` values contain the
absolute AB magnitude of the sun in Johnson/Cousins bands from [1]_.

References
----------
.. [1] Christopher N. A. Willmer 2018 ApJS 236 47

'''


def luminosity_from_absolute_magnitude(absolute_magnitude, zeropoint=None):
    """Converts absolute magnitudes to luminosities.

    Parameters
    ----------
    absolute_magnitude : array_like
        Input absolute magnitudes.
    zeropoint : float or str, optional
        Zeropoint for the conversion. If a string is given, uses the reference
        luminosities from `luminosity_in_band`.

    Returns
    -------
    luminosity : array_like
        Luminosity values.

    """

    if zeropoint is None:
        zeropoint = 0.
    elif isinstance(zeropoint, str):
        if zeropoint not in luminosity_in_band:
            raise KeyError('unknown zeropoint `{}`'.format(zeropoint))
        zeropoint = -luminosity_in_band[zeropoint]

    return 10.**(-0.4*np.add(absolute_magnitude, zeropoint))


def absolute_magnitude_from_luminosity(luminosity, zeropoint=None):
    """Converts luminosities to absolute magnitudes.

    Parameters
    ----------
    luminosity : array_like
        Input luminosity.
    zeropoint : float, optional
        Zeropoint for the conversion. If a string is given, uses the reference
        luminosities from `luminosity_in_band`.

    Returns
    -------
    absolute_magnitude : array_like
        Absolute magnitude values.

    """

    if zeropoint is None:
        zeropoint = 0.
    elif isinstance(zeropoint, str):
        if zeropoint not in luminosity_in_band:
            raise KeyError('unknown zeropoint `{}`'.format(zeropoint))
        zeropoint = -luminosity_in_band[zeropoint]

    return -2.5*np.log10(luminosity) - zeropoint
