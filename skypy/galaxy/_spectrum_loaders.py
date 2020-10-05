'''Implementations for spectrum loaders.'''

import numpy as np
import specutils
import astropy.utils.data
import astropy.table
from astropy import __version__ as astropy_version
from astropy import units

import os
import urllib
from pkg_resources import resource_filename


def download_file(url, cache=True):
    '''download_file with some specific settings'''
    if astropy_version.startswith('3.'):  # pragma: no cover
        extra_kwargs = {}
    else:
        extra_kwargs = {'pkgname': 'skypy'}
    return astropy.utils.data.download_file(
            url, cache=cache, show_progress=False, **extra_kwargs)


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
    '''load data from the skypy data package'''

    # result is spectrum or list of spectra
    spectra = None

    # load each tag separately
    for tag in tags:

        # get resource filename from module, name, and tag
        filename = resource_filename(f'skypy-data.{module}', f'{name}_{tag}.ecsv')

        # load the data file
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
    throughput = np.squeeze(throughput)*units.dimensionless_unscaled

    # return the bandpasses as Spectrum1D
    return specutils.Spectrum1D(spectral_axis=spectral_axis, flux=throughput)


spectrum_loaders = [
    # bandpasses
    ('(Johnson)_(U)?(B)?(V)?', skypy_data_loader, 'bandpasses'),
    ('(Cousins)_(R)?(I)?', skypy_data_loader, 'bandpasses'),
    ('DECam_(g)?(r)?(i)?(z)?(Y)?', decam_loader),

    # spectrum templates
    ('(kcorrect)_((?:raw)?spec(?:_nl)?(?:_nd)?)', skypy_data_loader, 'spectrum_templates'),

    # catchall file loader
    ('(.*)', file_loader),
]
