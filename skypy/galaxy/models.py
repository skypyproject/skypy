def _herbel_params(gal_type):
    """ Returns the parameters defined by Herbel et al. (2017)

    Parameters
    ----------
    gal_type : str
        Predefined galaxy type. Has to be 'blue' or 'red'

    Returns
    -------
    tuple
    The model parameters.
    """
    if gal_type == 'red':
        a_m = -0.70798041
        b_m = -20.37196157
        a_phi = -0.70596888
        b_phi = 0.0035097
        alpha = -0.5
    elif gal_type == 'blue':
        a_m = -0.9408582
        b_m = -20.40492365
        a_phi = -0.10268436
        b_phi = 0.00370253
        alpha = -1.3
    else:
        raise ValueError('Galaxy type has to be blue or red.')

    return a_m, b_m, a_phi, b_phi, alpha
