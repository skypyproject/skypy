'''decorators for extended function definitions'''


import numpy as np

from inspect import signature
from functools import wraps
from astropy.cosmology import default_cosmology


__all__ = [
    'broadcast_arguments',
    'dependent_argument',
    'uses_default_cosmology',
]


def broadcast_arguments(*broadcast_args):
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
    sig = signature(function)

    @wraps(function)
    def wrapper(*args, **kwargs):
        given = sig.bind_partial(*args, **kwargs)
        if 'cosmology' not in given.arguments:
            given.arguments['cosmology'] = default_cosmology.get()
        return function(*given.args, **given.kwargs)

    return wrapper
