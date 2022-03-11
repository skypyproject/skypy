"""Models of galaxy magnitude errors.
"""

import numpy as np


__all__ = [
    'rykoff_error',
]


def rykoff_error(magnitude, magnitude_limit, magnitude_zp, a, b, error_limit=None):
    r"""Magnitude error acoording to the model from Rykoff et al. (2015).

    Given the apparent magnitude of a galaxy calculate the magnitude error that is introduced
    by the survey specifications and follows the model described in [1]_.

    Parameters
    ----------
    magnitude: array_like
        Galaxy apparent magnitude. This and the other array_like parameters must
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
        The apparent magnitude error in the Rykoff et al. (2018) model. This is a scalar
        if magnitude, magnitude_limit, magnitude_zp, a and b are scalars.

    Notes
    -----
    Rykoff et al. (2018) (see [1]_) describe the error of the apparent magnitude :math:`m` as

    .. math::

        \sigma_m(m;m_{\mathrm{lim}}, t_{\mathrm{eff}}) &=
            \sigma_m(F(m);F_{\mathrm{noise}}(m_{\mathrm{lim}}), t_{\mathrm{eff}}) \\
        &= \frac{2.5}{\ln(10)} \left[ \frac{1}{Ft_{\mathrm{eff}}}
            \left( 1 + \frac{F_{\mathrm{noise}}}{F} \right) \right]^{1/2} \;,

    where

    .. math::

        F=10^{-0.4(m - m_{\mathrm{ZP}})}

    is the galaxy's flux,

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

    References
    ----------
    .. [1] Rykoff E. S., Rozo E., Keisler R., 2015, eprint arXiv:1509.00870

    """

    flux = 10 ** (-0.4 * (np.subtract(magnitude, magnitude_zp)))
    flux_limit = 10 ** (-0.4 * (np.subtract(magnitude_limit, magnitude_zp)))
    t_eff = np.exp(a + b * (np.subtract(magnitude_limit, 21.0)))
    flux_noise = flux_limit ** 2 * t_eff / 10 ** 2 - flux_limit
    error = 2.5 / np.log(10) * (1 / (flux * t_eff) * (1 + flux_noise / flux)) ** 0.5

    # replace the values larger than the error limit with the error_limit
    if error_limit is not None:
        error = np.where(error > error_limit, error_limit, error)

    return error
