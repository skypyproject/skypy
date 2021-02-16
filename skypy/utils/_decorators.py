'''decorators for extended function definitions'''


import numpy as np

from inspect import signature
from functools import wraps


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
