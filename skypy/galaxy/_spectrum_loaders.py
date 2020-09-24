'''Implementations for spectrum loaders.'''

import numpy as np
import specutils
import astropy.utils.data
import astropy.table
from astropy import __version__ as astropy_version
from astropy import units

import os
import urllib


def download_file(url, cache=True):
    '''download_file with some specific settings'''
    if astropy_version.startswith('3.'):  # pragma: no cover
        extra_kwargs = {}
    else:
        extra_kwargs = {'pkgname': 'skypy'}
    try:
        filename = astropy.utils.data.download_file(
                url, cache=cache, show_progress=False, **extra_kwargs)
    except urllib.error.URLError:  # pragma: no cover
        raise FileNotFoundError('data file not available: {}'.format(url))
    return filename


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
        if (np.ndim(flux_a) == np.ndim(flux_b)
                and np.shape(flux_a)[1:] == np.shape(flux_b)[1:]):
            return specutils.Spectrum1D(spectral_axis=a.spectral_axis,
                                        flux=np.vstack([flux_a, flux_b])*a.flux.unit)

    return specutils.SpectrumList([a, b])


def file_loader(*filenames):
    '''load a file'''
    spectra = []
    for filename in filenames:
        spectra.append(specutils.Spectrum1D.read(filename))
    return spectra[0] if len(spectra) == 1 else specutils.SpectrumList(spectra)


def skypy_data_loader(folder, name, *tags):
    '''load data from the skypy data repository'''

    # move most of these to config?
    request = {
        'repo': 'https://github.com/skypyproject/data',
        'version': '1.0',
        'folder': urllib.parse.quote_plus(folder),
        'name': urllib.parse.quote_plus(name),
        'format': 'ecsv',
    }

    # result is spectrum or list of spectra
    spectra = None

    # load each tag separately
    for tag in tags:

        # build url from request dict and tag
        url = '{repo:}/raw/{version:}/{folder:}/{name:}.{tag:}.{format:}'.format(
                **request, tag=urllib.parse.quote_plus(tag))

        # download file from remote server to cache
        filename = download_file(url)

        # load the data file
        data = astropy.table.Table.read(filename, format='ascii.ecsv')

        # get the spectral axis
        spectral_axis = data['spectral_axis'].quantity

        # load all templates
        flux_unit = data['flux_0'].unit
        fluxes = []
        while 'flux_%d' % len(fluxes) in data.colnames:
            fluxes.append(data['flux_%d' % len(fluxes)].quantity.to_value(flux_unit))
        fluxes = fluxes*flux_unit

        # construct the Spectrum1D
        spectrum = specutils.Spectrum1D(spectral_axis=spectral_axis, flux=fluxes)

        # combine with existing
        spectra = combine_spectra(spectra, spectrum)

    return spectra


def decam_loader(*bands):
    '''load DECam bandpass filters'''

    # download DECam filter data
    filename = download_file(
            'http://www.ctio.noao.edu/noao/sites/default/files/DECam/STD_BANDPASSES_DR1.fits')

    # load the data file
    data = astropy.table.Table.read(filename, format='fits')

    # set units
    data['LAMBDA'].unit = units.angstrom

    # get the spectral axis
    spectral_axis = data['LAMBDA'].quantity

    # load requested bands
    throughput = []
    for band in bands:
        throughput.append(data[band])
    throughput = throughput*units.dimensionless_unscaled

    # return the bandpasses as Spectrum1D
    return specutils.Spectrum1D(spectral_axis=spectral_axis, flux=throughput)


spectrum_loaders = [
    # bandpasses
    ('(Johnson)_(U)?(B)?(V)?', skypy_data_loader, 'bandpasses'),
    ('(Cousins)_(R)?(I)?', skypy_data_loader, 'bandpasses'),
    ('DECam_(g)?(r)?(i)?(z)?(Y)?', decam_loader),

    # spectrum templates
    ('(kcorrect)_([a-z]+)', skypy_data_loader, 'spectrum_templates'),

    # catchall file loader
    ('(.*)', file_loader),
]
