from astropy.table import Table


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
        function_name = config.get('function')
        module = __import__(module_name, fromlist=function_name)
        function = getattr(module, function_name)

        # Parse arguments
        args = config.get('args', [])
        kwargs = config.get('kwargs', {})
        req = config.get('requires', {})
        req = {k: getattr(self, v) for k, v in req.items()}

        # Call function
        return function(*args, **kwargs, **req)
