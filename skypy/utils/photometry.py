"""Photometry module.

"""


from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.special


__all__ = [
    'absolute_magnitude_from_luminosity',
    'luminosity_from_absolute_magnitude',
    'luminosity_in_band',
    'mag_ab',
    'SpectrumTemplates',
    'magnitude_error_rykoff',
    'logistic_completeness_function',
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


def magnitude_error_rykoff(magnitude, magnitude_limit, magnitude_zp, a, b, error_limit=np.inf):
    r"""Magnitude error acoording to the model from Rykoff et al. (2015).

    Given an apparent magnitude calculate the magnitude error that is introduced
    by the survey specifications and follows the model described in Rykoff et al. (2015).

    Parameters
    ----------
    magnitude: array_like
        Apparent magnitude. This and the other array_like parameters must
        be broadcastable to the same shape.
    magnitude_limit: array_like
        :math:`10\sigma` limiting magnitude of the survey. This and the other
        array_like parameters must be broadcastable to the same shape.
    magnitude_zp: array_like
        Zero-point magnitude of the survey. This and the other array_like parameters must
        be broadcastable to the same shape.
    a,b: array_like
        Model parameters: a is the intercept and
        b is the slope of the logarithmic effective time.
        These and the other array_like parameters must be broadcastable to the same shape.
    error_limit: float, optional
        Upper limit of the returned error. If given, all values larger than this value
        will be set to error_limit. Default is None.

    Returns
    -------
    error: ndarray
        The apparent magnitude error in the Rykoff et al. (2015) model. This is a scalar
        if magnitude, magnitude_limit, magnitude_zp, a and b are scalars.

    Notes
    -----
    Rykoff et al. (2015) (see [1]_) describe the error of the apparent magnitude :math:`m` as

    .. math::

        \sigma_m(m;m_{\mathrm{lim}}, t_{\mathrm{eff}}) &=
            \sigma_m(F(m);F_{\mathrm{noise}}(m_{\mathrm{lim}}), t_{\mathrm{eff}}) \\
        &= \frac{2.5}{\ln(10)} \left[ \frac{1}{Ft_{\mathrm{eff}}}
            \left( 1 + \frac{F_{\mathrm{noise}}}{F} \right) \right]^{1/2} \;,

    where

    .. math::

        F=10^{-0.4(m - m_{\mathrm{ZP}})}

    is the source's flux,

    .. math::

        F_\mathrm{noise} = \frac{F_{\mathrm{lim}}^2 t_{\mathrm{eff}}}{10^2} - F_{\mathrm{lim}}

    is the effective noise flux and :math:`t_\mathrm{eff}` is the effective exposure time
    (we absorbed the normalisation constant :math:`k` in the definition of
    :math:`t_\mathrm{eff}`).
    Furthermore, :math:`m_\mathrm{ZP}` is the zero-point magnitude of the survey and
    :math:`F_\mathrm{lim}` is the :math:`10\sigma` limiting flux.
    Accordingly, :math:`m_\mathrm{lim}` is the :math:`10\sigma` limiting magnitud
    associated with :math:`F_\mathrm{lim}`.

    The effective exposure time is described by

    .. math::

        \ln{t_\mathrm{eff}} = a + b(m_\mathrm{lim} - 21)\;,

    where :math:`a` and :math:`b` are free parameters.

    Further note that the model was originally used for SDSS galaxy photometry.

    References
    ----------
    .. [1] Rykoff E. S., Rozo E., Keisler R., 2015, eprint arXiv:1509.00870

    """

    flux = luminosity_from_absolute_magnitude(magnitude, -magnitude_zp)
    flux_limit = luminosity_from_absolute_magnitude(magnitude_limit, -magnitude_zp)
    t_eff = np.exp(a + b * np.subtract(magnitude_limit, 21.0))
    flux_noise = np.square(flux_limit / 10) * t_eff - flux_limit
    error = 2.5 / np.log(10) * np.sqrt((1 + flux_noise / flux) / (flux * t_eff))

    return np.minimum(error, error_limit)


def logistic_completeness_function(magnitude, magnitude_95, magnitude_50):
    r'''Logistic completeness function.

    This function calculates the logistic completeness function (based on eq. (7) in
    Lopez-Sanjuan C. et al. (2017))

    .. math::

        p(m) = \frac{1}{1 + \exp[\kappa (m - m_{50})]}\;,

    which describes the probability :math:`p(m)` that an object of magnitude :math:`m` is detected
    in a specific band and with

    .. math::

        \kappa = \frac{\ln(\frac{1}{19})}{m_{95} - m_{50}}\;.

    Here, :math:`m_{95}` and :math:`m_{50}` are the 95% and 50% completeness
    magnitudes, respectively.

    Parameters
    ----------
    magnitude : array_like
        Magnitudes. Can be multidimensional for computing with multiple filter bands.
    magnitude_95 : scalar or 1-D array_like
        95% completeness magnitude.
        If `magnitude_50` is 1-D array it has to be scalar or 1-D array of the same shape.
    magnitude_50 : scalar or 1-D array_like
        50% completeness magnitude.
        If `magnitude_95` is 1-D array it has to be scalar or 1-D array of the same shape.

    Returns
    -------
    probability : scalar or array_like
        Probability of detecting an object with magnitude :math:`m`.
        Returns array_like of the same shape as magnitude.
        Exemption: If magnitude is scalar and `magnitude_95` or `magnitude_50`
        is array_like of shape (nb, ) it returns array_like of shape (nb, ).

    References
    -----------
    .. [1] Lopez-Sanjuan C. et al., 2017, A&A, 599, A62

    '''

    kappa = np.log(1. / 19) / np.subtract(magnitude_95, magnitude_50)
    arg = kappa * np.subtract(magnitude, magnitude_50)
    return scipy.special.expit(-arg)
