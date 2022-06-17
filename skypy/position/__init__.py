'''Module for sampling positions on the sky.'''

__all__ = []

from ._uniform import uniform_around, uniform_in_pixel

__all__ += [
    'uniform_around',
    'uniform_in_pixel',
]
