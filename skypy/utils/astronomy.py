import numpy as np


def convert_abs_mag_to_lum(absolute_magnitude):
    """ Converts absolute magnitudes into luminosities

    Parameters
    ----------
    absolute_magnitude : array_like
                    Input absolute magnitudes
    Returns
    -------
    ndarray, or float if input is scalar
    Luminosity values.
    """

    return 10 ** (-0.4 * absolute_magnitude)


def convert_lum_to_abs_mag(luminosity):
    """

    Parameters
    ----------
    luminosity : array_like
            Input luminosity

    Returns
    -------
    ndarray, or float if input is scalar
    Absolute magnitude values
    """
    return -np.log(luminosity)/(0.4 * np.log(10))
