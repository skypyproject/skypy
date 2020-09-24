'''decorators for extended function definitions'''


import numpy as np

from inspect import signature
from functools import wraps
from astropy import units
from astropy.cosmology import default_cosmology


def broadcast_arguments(*broadcast_args):
    '''Decorator that broadcasts arguments.

    Parameters
    ----------
    broadcast_args : tuple
        The names of the decorated function arguments to be broadcast together
        using numpy.broadcast_arrays.
    '''
    def decorator(function):
        sig = signature(function)
        for arg in broadcast_args:
            if arg not in sig.parameters:
                raise ValueError('@broadcast_arguments: '
                                 'unknown argument `{}` in function `{}`'
                                 .format(arg, function.__qualname__))

        @wraps(function)
        def wrapper(*args, **kwargs):
            given = sig.bind(*args, **kwargs)
            bc = np.broadcast_arrays(*map(given.arguments.get, broadcast_args))
            given.arguments.update(dict(zip(broadcast_args, bc)))
            return function(*given.args, **given.kwargs)

        return wrapper
    return decorator


def dependent_argument(dependent_arg, *independent_args):
    '''Decorator to evaluate a dependent argument.

    Parameters
    ----------
    dependent_arg : str
        The name of the decorated function's dependent argument that can
        optionally be passed as a callable object to be evaluated.
    independent_args : tuple
        The names of the decorated function's independent arguments to be
        passed as function arguments when evaluating the dependent argument.
    '''
    def decorator(function):
        sig = signature(function)
        for arg in [dependent_arg, *independent_args]:
            if arg not in sig.parameters:
                raise ValueError('@dependent_argument: '
                                 'unknown argument `{}` in function `{}`'
                                 .format(arg, function.__qualname__))

        @wraps(function)
        def wrapper(*args, **kwargs):
            given = sig.bind(*args, **kwargs)
            f = given.arguments[dependent_arg]
            if callable(f):
                given.arguments[dependent_arg] \
                        = f(*map(given.arguments.get, independent_args))
            return function(*given.args, **given.kwargs)

        return wrapper
    return decorator


def uses_default_cosmology(function):
    '''Decorator to use the Astropy default cosmology if none is given.'''
    sig = signature(function)

    @wraps(function)
    def wrapper(*args, **kwargs):
        given = sig.bind_partial(*args, **kwargs)
        if 'cosmology' not in given.arguments:
            given.arguments['cosmology'] = default_cosmology.get()
        return function(*given.args, **given.kwargs)

    return wrapper


def spectral_data_input(**parameters):
    '''Decorator to load spectral data automatically and validate units.

    Keyword arguments are pairs of parameters and their required `flux` units.

    Examples
    --------
    >>> from astropy import units
    >>> from skypy.utils import spectral_data_input

    This function combines two bandpasses (i.e. dimensionless `flux` units):
    >>> @spectral_data_input(bp1=units.dimensionless_unscaled, bp2=units.dimensionless_unscaled)
    ... def combine_bandpasses(bp1, bp2):
    ...     return bp1*bp2

    '''
    def decorator(function):
        sig = signature(function)
        for par in parameters:
            if par not in sig.parameters:
                raise ValueError('@spectral_data_input: '
                                 'unknown parameter `{}` for function `{}`'
                                 .format(par, function.__qualname__))

        @wraps(function)
        def wrapper(*args, **kwargs):
            given = sig.bind(*args, **kwargs)
            for par, unit in parameters.items():
                arg = given.arguments[par]
                if (isinstance(arg, str)
                        or hasattr(arg, '__iter')
                        and all(isinstance(elem, str) for elem in arg)):
                    # import here to prevent circular import
                    from ..galaxy.spectrum import load_spectral_data
                    arg = load_spectral_data(arg)
                    given.arguments[par] = arg
                if not arg.unit.is_equivalent(
                        unit, equivalencies=units.spectral_density(arg.spectral_axis)):
                    raise units.UnitConversionError(
                            '{} does not have {} units'.format(par, unit.physical_type))
            return function(*given.args, **given.kwargs)

        return wrapper
    return decorator
