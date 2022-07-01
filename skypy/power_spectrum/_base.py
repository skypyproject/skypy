from abc import ABCMeta, abstractmethod
from inspect import signature
import numpy as np
from scipy.interpolate import RectBivariateSpline


class PowerSpectrum(metaclass=ABCMeta):
    '''Base class for power spectrum calculation'''

    @abstractmethod
    def __init__(self):
        raise NotImplementedError


class TabulatedPowerSpectrum(PowerSpectrum):
    '''Base class for power spectrum interpolation'''

    def __init__(self, wavenumber, redshift, power_spectrum):
        kx = min(np.size(redshift)-1, signature(RectBivariateSpline).parameters['kx'].default)
        ky = min(np.size(wavenumber)-1, signature(RectBivariateSpline).parameters['ky'].default)
        self.interpolator = RectBivariateSpline(redshift, np.log10(wavenumber),
                                                np.log10(power_spectrum), kx=kx, ky=ky)

    def __call__(self, wavenumber, redshift):
        shape = np.shape(redshift) + np.shape(wavenumber)
        sort_k = np.argsort(wavenumber)
        sort_z = np.argsort(redshift)
        unsort_k = np.argsort(sort_k)
        unsort_z = np.argsort(sort_z)
        interp = self.interpolator(np.atleast_1d(redshift)[sort_z],
                                   np.log10(np.atleast_1d(wavenumber)[sort_k]))
        pzk = np.power(10, interp[unsort_z[:, np.newaxis], unsort_k]).reshape(shape)
        return pzk.item() if np.isscalar(wavenumber) and np.isscalar(redshift) else pzk
