import builtins
from importlib import import_module
import yaml
import re
from astropy.units import Quantity

__all__ = [
    'load_skypy_yaml',
]


def import_function(qualname):
    '''load function from fully qualified name'''
    path = qualname.split('.')
    module = builtins
    for i, key in enumerate(path[:-1]):
        if not hasattr(module, key):
            module = import_module('.'.join(path[:i+1]))
        else:
            module = getattr(module, key)
    function = getattr(module, path[-1])
    return function


class SkyPyLoader(yaml.SafeLoader):
    '''custom YAML loader class with SkyPy extensions'''

    @classmethod
    def load(cls, stream):
        '''load the first YAML document from stream'''
        loader = cls(stream)
        try:
            return loader.get_single_data()
        finally:
            loader.dispose()

    def construct_function(self, name, node):
        '''load function from !function tag

        tags are stored as a tuple `(function, args)`
        '''

        if isinstance(node, yaml.ScalarNode):
            args = self.construct_scalar(node)
        elif isinstance(node, yaml.SequenceNode):
            args = self.construct_sequence(node)
        elif isinstance(node, yaml.MappingNode):
            args = self.construct_mapping(node)

        try:
            function = import_function(name)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(f'{e}\n{node.start_mark}') from e

        return (function,) if args == '' else (function, args)

    def construct_quantity(self, node):
        value = self.construct_scalar(node)
        return Quantity(value)


# constructor for generic functions
SkyPyLoader.add_multi_constructor('!', SkyPyLoader.construct_function)

# constructor for quantities
SkyPyLoader.add_constructor('!quantity', SkyPyLoader.construct_quantity)
# Implicitly resolve quantities using the regex from astropy
SkyPyLoader.add_implicit_resolver('!quantity', re.compile(r'''
    \s*[+-]?((\d+\.?\d*)|(\.\d+)|([nN][aA][nN])|
    ([iI][nN][fF]([iI][nN][iI][tT][yY]){0,1}))([eE][+-]?\d+)?[.+-]? \w* \W+
''', re.VERBOSE), list('-+0123456789.'))


def load_skypy_yaml(filename):
    '''Read a SkyPy pipeline configuration from a YAML file.

    Parameters
    ----------
    filename : str
        The name of the configuration file.
    '''

    # read the YAML file
    with open(filename, 'r') as stream:
        return SkyPyLoader.load(stream) or {}
