from astropy.table import Table


class SkyPyDriver:

    def execute(self, config):

        # Cosmology
        self.cosmology = self.call_from_config(config['cosmology'])

        # Tables
        for table, columns in config['tables'].items():
            setattr(self, table, Table())
            for column, settings in columns.items():
                getattr(self, table)[column] = self.call_from_config(settings)

    def call_from_config(self, settings):

        # Import function
        module = __import__(settings['module'], fromlist=settings['function'])
        function = getattr(module, settings['function'])

        # Parse arguments
        args = settings.get('args', [])
        kwargs = settings.get('kwargs', {})
        req = settings.get('requires', {})
        req = {k: getattr(self, v) for k, v in req.items()}

        # Call function
        return function(*args, **kwargs, **req)
