import numpy as np


def luminosity_from_absolute_magnitude(absolute_magnitude):
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

    return np.power(10, -0.4*absolute_magnitude)


def absolute_magnitude_from_luminosity(luminosity):
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
