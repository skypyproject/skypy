from astropy.cosmology import default_cosmology
from astropy.table import Table


def import_function(module, function):

    fl = function.split('.')
    module = __import__(module, fromlist=fl[0])
    for f in fl:
        module = getattr(module, f)
    return module


class SkyPyDriver:

    def __init__(self):

        self.cosmology = default_cosmology.get()

    def execute(self, config):

        for table, columns in config['tables'].items():
            t = self.new_table(table)
            for col, settings in columns.items():
                module_name = settings['module']
                function_name = settings['function']
                function = import_function(module_name, function_name)
                args = settings.get('args', [])
                kwargs = settings.get('kwargs', {})
                req = settings.get('requires', {})
                req = {k: self.get(v) for k, v in req.items()}
                t[col] = function(*args, **kwargs, **req)

    def new_table(self, name):

        setattr(self, name, Table())
        return getattr(self, name)

    def get(self, name):

        attribute, *key = name.split('.')
        attribute = getattr(self, attribute)
        if type(attribute) == Table:
            attribute = attribute[key]
        return attribute
