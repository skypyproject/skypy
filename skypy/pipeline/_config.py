import builtins
from importlib import import_module
import yaml

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


def function_tag(loader, name, node):
    '''load function from !function tag

    tags are stored as a tuple `(function, args)`
    '''

    if isinstance(node, yaml.ScalarNode):
        args = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        args = loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode):
        args = loader.construct_mapping(node)

    try:
        function = import_function(name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f'{e}\n{node.start_mark}') from e

    return (function,) if args == '' else (function, args)


def load_skypy_yaml(filename):
    '''Read a SkyPy pipeline configuration from a YAML file.

    Parameters
    ----------
    filename : str
        The name of the configuration file.
    '''

    yaml.SafeLoader.add_multi_constructor('!', function_tag)
    with open(filename, 'r') as stream:
        return yaml.safe_load(stream) or {}
