import yaml

__all__ = [
    'skypy_config',
]


def _yaml_tag(loader, tag, node):
    '''handler for generic YAML tags

    tags are stored as a tuple `(tag, value)`
    '''

    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode):
        value = loader.construct_mapping(node)

    # tags without arguments have empty string value
    if value == '':
        return tag,

    return tag, value


def skypy_config(filename):
    '''Read a SkyPy pipeline configuration from a YAML file.

    Parameters
    ----------
    filename : str
        The name of the configuration file.
    '''

    yaml.SafeLoader.add_multi_constructor('!', _yaml_tag)
    with open(filename, 'r') as stream:
        return yaml.safe_load(stream) or {}
