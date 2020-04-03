# patch scipy-specific strings in generated docstrings

import sys


# list of exported symbols
__all__ = [
    'rv_example',
]


_EXAMPLE_IMPORT = 'from %(module)s import %(name)s\n'
_EXAMPLE_ARGS = '>>> %(shapes)s = %(args)s\n'


def rv_example(*example_args):
    '''Decorator for random variables with default example docstring.'''

    example_args = ', '.join('%.3g' % arg for arg in example_args)

    def decorator_rv_example(rv_gen):
        if sys.flags.optimize < 2:
            init = rv_gen.__init__

            def init_and_patch(self, *args, **kwargs):
                init(self, *args, **kwargs)

                self.__doc__ = self.__doc__.replace(
                    _EXAMPLE_IMPORT % {'module': 'scipy.stats',
                                       'name': self.name},
                    _EXAMPLE_IMPORT % {'module': 'skypy.stats',
                                       'name': self.name}
                ).replace(
                    _EXAMPLE_ARGS % {'shapes': self.shapes,
                                     'args': ''},
                    _EXAMPLE_ARGS % {'shapes': self.shapes,
                                     'args': example_args}
                )

            rv_gen.__init__ = init_and_patch

        return rv_gen

    return decorator_rv_example
