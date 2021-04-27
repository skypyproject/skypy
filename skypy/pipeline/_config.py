import builtins
from importlib import import_module
import yaml
import re
from astropy.units import Quantity
from ._items import Ref, Call

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

    def construct_mapping(self, node, deep=False):
        mapping = super().construct_mapping(node, deep)
        for key in mapping:
            if not isinstance(key, str):
                raise ValueError(f'key "{key}" is of non-string type "{type(key).__name__}"\n'
                                 f'{node.start_mark}')
        return mapping

    def construct_ref(self, node):
        ref = self.construct_scalar(node)
        if ref[:1] == '$':
            ref = ref[1:]
        if not ref:
            raise ValueError(f'empty reference\n{node.start_mark}')
        return Ref(ref)

    def construct_call(self, name, node):
        try:
            object = import_function(name)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(f'{e}\n{node.start_mark}') from e

        if isinstance(node, yaml.ScalarNode):
            if node.value:
                raise ValueError(f'{node.value}: ScalarNode should be empty to import an object')
            return object
        else:
            if isinstance(node, yaml.SequenceNode):
                args = self.construct_sequence(node)
                kwargs = {}
            if isinstance(node, yaml.MappingNode):
                args = []
                kwargs = self.construct_mapping(node)
            return Call(object, args, kwargs)

    def construct_quantity(self, node):
        value = self.construct_scalar(node)
        return Quantity(value)


# constructor for references
SkyPyLoader.add_constructor('!ref', SkyPyLoader.construct_ref)
# implicitly resolve $references
SkyPyLoader.add_implicit_resolver('!ref', re.compile(r'\$\w+'), ['$'])

# constructor for generic function calls
SkyPyLoader.add_multi_constructor('!', SkyPyLoader.construct_call)

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
