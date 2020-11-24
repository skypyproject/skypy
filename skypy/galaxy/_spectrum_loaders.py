'''Implementations for spectrum loaders.'''

import numpy as np
import astropy.utils.data
import astropy.table
from astropy import units

import os
import urllib
from pkg_resources import resource_filename

# this file is only ever imported when specutils is present
# but without the try/except pytest will fail when doctests are discovered
try:
    import specutils
except ImportError:
    pass

# this file is only ever imported when speclite is present
# but without the try/except pytest will fail when doctests are discovered
try:
    import speclite.filters
except ImportError:
    pass


def combine_spectra(a, b):
    '''combine two spectra'''
    if a is None or b is None:
        return a or b

    if isinstance(a, specutils.SpectrumList) or isinstance(b, specutils.SpectrumList):
        a = a if isinstance(a, specutils.SpectrumList) else specutils.SpectrumList([a])
        b = b if isinstance(b, specutils.SpectrumList) else specutils.SpectrumList([b])
        return specutils.SpectrumList(a + b)

    if (len(a.spectral_axis) == len(b.spectral_axis)
            and np.allclose(a.spectral_axis, b.spectral_axis, atol=0, rtol=1e-10)
            and a.flux.unit.is_equivalent(b.flux.unit)):
        flux_a = np.atleast_2d(a.flux.value)
        flux_b = np.atleast_2d(b.flux.to_value(a.flux.unit))
        if flux_a.shape[1:] == flux_b.shape[1:]:
            return specutils.Spectrum1D(spectral_axis=a.spectral_axis,
                                        flux=np.concatenate([flux_a, flux_b])*a.flux.unit)

    return specutils.SpectrumList([a, b])


def file_loader(*filenames):
    '''load a file'''
    spectra = []
    for filename in filenames:
        spectra.append(specutils.Spectrum1D.read(filename))
    return spectra[0] if len(spectra) == 1 else specutils.SpectrumList(spectra)


def skypy_data_loader(module, name, *tags):
    '''Load skypy module data'''

    # result is spectrum or list of spectra
    spectra = None

    # load each tag separately
    for tag in tags:

        # load from skypy/data
        filename = resource_filename('skypy', f'data/{module}/{name}_{tag}.ecsv.gz')
        data = astropy.table.Table.read(filename, format='ascii.ecsv')

        # get the spectral axis
        spectral_axis = data['spectral_axis'].quantity

        # load all templates
        flux_unit = data['flux_0'].unit
        fluxes = []
        while 'flux_%d' % len(fluxes) in data.colnames:
            fluxes.append(data['flux_%d' % len(fluxes)].quantity.to_value(flux_unit))
        fluxes = np.squeeze(fluxes)*flux_unit

        # construct the Spectrum1D
        spectrum = specutils.Spectrum1D(spectral_axis=spectral_axis, flux=fluxes)

        # combine with existing
        spectra = combine_spectra(spectra, spectrum)

    return spectra


def speclite_loader(name, *bands):
    '''load data from the speclite package'''

    # result is spectrum or list of spectra
    spectra = None

    # load each band sperately
    for band in bands:

        # get resource filename from name and bands
        filter_name = f'{name}-{band}'

        # load the filter response as a speclite.FilterResponse object
        filter_object = speclite.filters.load_filter(filter_name)

        # get the spectral axis as astropy quantity
        spectral_axis = filter_object.wavelength*speclite.filters.default_wavelength_unit

        # get throughput
        throughput = filter_object.response * units.dimensionless_unscaled

        # construct the Spectrum1D
        spectrum = specutils.Spectrum1D(spectral_axis=spectral_axis, flux=throughput)

        # combine with existing
        spectra = combine_spectra(spectra, spectrum)

    return spectra


spectrum_loaders = [
    # bandpasses
    ('(decam2014)_(u)?(g)?(r)?(i)?(z)?(Y)?', speclite_loader),
    ('(sdss2010)_(u)?(g)?(r)?(i)?(z)?', speclite_loader),
    ('(wise2010)_(W1)?(W2)?(W3)?(W4)?', speclite_loader),
    ('(hsc2017)_(g)?(r)?(i)?(z)?(y)?', speclite_loader),
    ('(lsst2016)_(u)?(g)?(r)?(i)?(z)?(y)?', speclite_loader),
    ('(bessell)_(U)?(B)?(V)?(R)?(I)?', speclite_loader),
    ('(BASS)_(g)?(r)?', speclite_loader),
    ('(MzLS)_(z)?', speclite_loader),

    # spectrum templates
    ('(kcorrect)_((?:raw)?spec(?:_nl)?(?:_nd)?)', skypy_data_loader, 'spectrum_templates'),

    # catchall file loader
    ('(.*)', file_loader),
]
