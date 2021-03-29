from abc import ABCMeta, abstractmethod
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
        self.interpolator = RectBivariateSpline(redshift, np.log10(wavenumber),
                                                np.log10(power_spectrum))

    def __call__(self, wavenumber, redshift):
        shape = np.shape(redshift) + np.shape(wavenumber)
        return np.power(10, self.interpolator(redshift, np.log10(wavenumber))).reshape(shape)
