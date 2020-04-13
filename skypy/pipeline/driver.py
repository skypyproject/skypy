"""Driver module.

This module provides methods to run pipelines of functions with dependencies
and handle their results.

.. autosummary::
   :nosignatures:
   :toctree: ../api/

   SkyPyDriver

"""

from astropy.table import Table
import networkx
import re


class SkyPyDriver:
    r'''Class for running pipelines.

    This is the main class for running pipelines of functions with dependencies
    and using their results to generate tables.
    '''

    def execute(self, config, file_format=None):
        r'''Run a pipeline.

        This function runs a pipeline of functions to generate the cosmology
        and the columns of a set of tables. It uses a Directed Acyclic Graph to
        determine a non-blocking order of execution that resolves any
        dependencies, see [1]_. Tables can optionally be written to file.

        Parameters
        ----------
        config : dict-like
            Configuration for the pipeline.
        file_format : str
            File format used to write tables. Files are written using the
            Astropy unified file read/write interface; see [2]_ for supported
            file formats. If None (default) tables are not written to file.

        Notes
        -----
        Each step in the pipeline is configured by a dictionary specifying:

        - 'function' : the name of the function
        - 'module' : the name of the the module to import 'function' from
        - 'args' : a list of positional arguments (by value)
        - 'kwargs' : a dictionary of keyword arguments
        - 'requires' : a dictionary of keyword arguments

        Note that 'kwargs' specifices keyword arguments by value, wheras
        'requires' specifices the names of previous steps in the pipeline and
        uses their return values as keyword arguments.

        'config' should contain 'cosmology' and/or 'tables'. 'cosmology' should
        return a dictionary configuring a function that returns an
        astropy.cosmology.Cosmology object. 'tables' should contain a set of
        nested dictionaries, first giving the names of each table, then the
        names of each column within each table. Each column should return a
        dictionary configuring a function that returns an array-like object.

        See [3]_ for examples of pipeline configurations in yaml format.

        References
        ----------
        .. [1] https://networkx.github.io/documentation/stable/
        .. [2] https://docs.astropy.org/en/stable/io/unified.html
        .. [3] https://github.com/skypyproject/skypy/tree/master/examples
        '''

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
