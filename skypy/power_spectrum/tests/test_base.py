import numpy as np


def test_TabulatedPowerSpectrum():
    from skypy.power_spectrum import TabulatedPowerSpectrum

    def power_spectrum_dummy(wavenumber, redshift):
        # returns log10 power spectrum on redshift wavelength grid -> 2D
        redshift = redshift.reshape(50, 1)
        wavenumber = wavenumber.reshape(1, 100)
        return 0.4*redshift + 0.2*np.log10(wavenumber) + 2

    wavenumber = np.linspace(800, 9000, 100)
    redshift = np.linspace(0, 3, 50)
    power_spectrum = np.power(10, power_spectrum_dummy(wavenumber, redshift))
    tab_power_spectrum = TabulatedPowerSpectrum(wavenumber, redshift, power_spectrum)

    wavenumber_test = np.linspace(1100, 7000, 100)
    redshift_test = np.linspace(1, 2, 50)
    interp_power = tab_power_spectrum(wavenumber_test, redshift_test)
    power_spectrum_test = np.power(10, power_spectrum_dummy(wavenumber_test, redshift_test))

    assert np.allclose(interp_power, power_spectrum_test)
