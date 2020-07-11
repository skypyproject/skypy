"""Driver module.

This module provides methods to run pipelines of functions with dependencies
and handle their results.
"""

import networkx
import re


__all__ = [
    'SkyPyDriver',
]


class SkyPyDriver:
    r'''Class for running pipelines.

    This is the main class for running pipelines of functions with dependencies
    and using their results to generate variables and tables.
    '''

    def execute(self, config, file_format=None):
        r'''Run a pipeline.

        This function runs a pipeline of functions to generate variables and
        the columns of a set of tables. It uses a Directed Acyclic Graph to
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

        'config' should contain the name and configuration of each variable
        and/or an entry named 'tables'. 'tables' should contain a set of nested
        dictionaries, first containing the name of each table, then the name
        and configuration of each column and optionally an entry named 'init'
        with a configuration that initialises the table. If 'init' is not
        specificed the table will be initialised as an empty astropy Table by
        default.

        See [3]_ for examples of pipeline configurations in YAML format.

        References
        ----------
        .. [1] https://networkx.github.io/documentation/stable/
        .. [2] https://docs.astropy.org/en/stable/io/unified.html
        .. [3] https://github.com/skypyproject/skypy/tree/master/examples
        '''

        # config contains settings for all variables and table initialisation
        # table_config contains settings for all table columns
        table_config = config.pop('tables', {})
        default_table = {'module': 'astropy.table', 'function': 'Table'}
        config.update({k: v.pop('init', default_table)
                      for k, v in table_config.items()})

        # Create a Directed Acyclic Graph of all jobs and dependencies
        dag = networkx.DiGraph()

        # Add nodes for each variable, table and column
        for job in config:
            dag.add_node(job)
        for table, columns in table_config.items():
            table_complete = '.'.join((table, "complete"))
            dag.add_node(table_complete)
            for column in columns.keys():
                job = '.'.join((table, column))
                dag.add_node(job)

        # Add edges for all requirements and dependencies
        for job, settings in config.items():
            dependencies = settings.get('depends', [])
            dependencies += settings.get('requires', {}).values()
            dag.add_edges_from((d, job) for d in dependencies)
        for table, columns in table_config.items():
            table_complete = '.'.join((table, "complete"))
            dag.add_edge(table, table_complete)
            for column, settings in columns.items():
                job = '.'.join((table, column))
                dag.add_edge(table, job)
                dag.add_edge(job, table_complete)
                dependencies = settings.get('depends', [])
                dependencies += settings.get('requires', {}).values()
                dag.add_edges_from((d, job) for d in dependencies)

        # Execute jobs in order that resolves dependencies
        for job in networkx.topological_sort(dag):
            if job.endswith('.complete'):
                continue
            elif job in config:
                settings = config.get(job)
                setattr(self, job, self._call_from_config(settings))
            else:
                table, column = job.split('.')
                settings = table_config[table][column]
                getattr(self, table)[column] = self._call_from_config(settings)

        # Write tables to file
        if file_format:
            for table in table_config.keys():
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
