from astropy.table import Table
import networkx
import re


class SkyPyDriver:

    def execute(self, config, file_format=None):

        # Create a Directed Acyclic Graph of all jobs and dependencies
        dag = networkx.DiGraph()
        if 'cosmology' in config:
            dag.add_node('cosmology')
        for table, columns in config.get('tables', {}).items():
            dag.add_node(table)
            for column, settings in columns.items():
                node = '.'.join((table, column))
                dag.add_node(node)
        for table, columns in config.get('tables', {}).items():
            for column, settings in columns.items():
                node = '.'.join((table, column))
                dag.add_edge(table, node)
                requirements = settings.get('requires', {}).values()
                dag.add_edges_from((r, node) for r in requirements)

        # Execute jobs in order that resolves dependencies
        for job in networkx.topological_sort(dag):
            if job == 'cosmology':
                settings = config.get('cosmology')
                self.cosmology = self._call_from_config(settings)
            elif job in config.get('tables', {}):
                setattr(self, job, Table())
            else:
                table, column = job.split('.')
                settings = config['tables'][table][column]
                getattr(self, table)[column] = self._call_from_config(settings)

        # Write tables to file
        if file_format:
            for table in config.get('tables', {}).keys():
                filename = '.'.join((table, file_format))
                getattr(self, table).write(filename)

    def _call_from_config(self, config):

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
