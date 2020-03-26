import numpy as np


def sampling_coefficients(redshift, a10, a20, a30, a40, a50,
                          a11, a21, a31, a41, a51):
    r""" Spectral coefficients to calculate the rest-frame spectral energy
        distribution of a galaxy following the Herbel et al. (2018) model.

    Parameters
    ----------
    redshift : array_like
        The redshift values of the galaxies for which the coefficients want to
        be sampled.
    a10, a20, a30, a40, a50, a11, a21, a31, a41, a51 : float or scalar
        Factors parameterising the Dirichlet distribution according to Equation
        (3.9) in [1].

    Returns
    -------
    coefficients : ndarray
        The spectral coefficients of the galaxies. The shape is (n, 5) with n
        the number of redshifts.

    Notes
    -------
    The rest-frame spectral energy distribution of galaxies can be written as a
    linear combination of the five kcorrect ([2]) template spectra :math:`f_i`
    (see [1])

    .. math::

        f(\lambda) = \sum_{i=1}^5 c_i f_i(\lambda) \;,

    where the coefficients :math:'c_i' were shown to follow a Dirichlet
    distribution of order 5. The five parameters describing the Dirichlet
    distribution are given by

    .. math::

        \alpha_i(z) = (\alpha_{i,0})^{1-z/z_1} \cdot (\alpha_{i,1})^{z/z_1} ;,.

    Here, :math:'\alpha_{i,0}' describes the galaxy population at redshift
    :math:'z=0' and :math:'\alpha_{i,1}' the population at :math:'z=z_1 > 0'.
    These parameters depend on the galaxy type and we chose :math:'z_1=1'.

    Examples
    -------
    >>> from skypy.galaxy.spectra import sampling_coefficients
    >>> import numpy as np

    Sample the coefficients according to [1] for n blue galaxies with redshifts
    between 0 and 1.

    >>> n = 100000
    >>> a10 = 2.079; a20 = 3.524; a30 = 1.917; a40 = 1.992; a50 = 2.536
    >>> a11 = 2.265; a21 = 3.862; a31 = 1.921; a41 = 1.685; a51 = 2.480
    >>> redshift = np.linspace(0,2, n)
    >>> coefficients = sampling_coefficients(redshift, a10, a20, a30, a40, a50,
    ...                                      a11, a21, a31, a41, a51)

    References
    -------
    [1] Herbel J., Kacprzak T., Amara A. et al., 2017, Journal of Cosmology and
    Astroparticle Physics, Issue 08, article id. 035 (2017)

    [2] Blanton M. R., Roweis S., 2007, The Astronomical Journal, Volume 133,
    Page 734
    """
    a1 = _spectral_coeff(redshift, a10, a11)
    a2 = _spectral_coeff(redshift, a20, a21)
    a3 = _spectral_coeff(redshift, a30, a31)
    a4 = _spectral_coeff(redshift, a40, a41)
    a5 = _spectral_coeff(redshift, a50, a51)

    if type(redshift) == float:
        redshift = np.array([redshift])

    a_vec = np.zeros(shape=(len(redshift), 5), dtype=float)
    a_vec[:, 0] = a1
    a_vec[:, 1] = a2
    a_vec[:, 2] = a3
    a_vec[:, 3] = a4
    a_vec[:, 4] = a5

    y = np.random.gamma(a_vec)
    sum_y = y.sum(1)
    coefficients = np.divide(y.T, sum_y.T).T
    return coefficients


def _spectral_coeff(z, ai0, ai1):
    return np.power(ai0, (1. - z / 1.)) * np.power(ai1, (z / 1.))
