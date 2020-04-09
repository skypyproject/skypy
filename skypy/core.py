from astropy.table import Table
import re


class SkyPyDriver:

    def execute(self, config):

        # Cosmology
        self.cosmology = self.call_from_config(config.get('cosmology'))

        # Tables
        for table, columns in config.get('tables', {}).items():
            setattr(self, table, Table())
            for column, settings in columns.items():
                getattr(self, table)[column] = self.call_from_config(settings)

    def call_from_config(self, config):

        # Import function
        module_name = config.get('module')
        object_name, function_name = re.search(r'^(\w*?)\.?(\w*)$',
                                               config.get('function')).groups()
        if object_name:
            module = __import__(module_name, fromlist=object_name)
            object = getattr(module, object_name)
            function = getattr(object, function_name)
        else:
            module = __import__(module_name, fromlist=function_name)
            function = getattr(module, function_name)

        # Parse arguments
        args = config.get('args', [])
        kwargs = config.get('kwargs', {})
        req = config.get('requires', {})
        req = {k: self.__getitem__(v) for k, v in req.items()}

        # Call function
        return function(*args, **kwargs, **req)

    def __getitem__(self, label):
        name, key = re.search(r'^(\w*)\.?(\w*)$', label).groups()
        item = getattr(self, name)
        return item[key] if key else item
