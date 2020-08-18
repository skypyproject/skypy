"""Driver module.

This module provides methods to run pipelines of functions with dependencies
and handle their results.
"""

from collections.abc import Mapping
from copy import deepcopy
from importlib import import_module
import builtins
import networkx
import re


__all__ = [
    'SkyPyDriver',
]


def _items(a):
    '''return keys and values for dict or list'''
    if hasattr(a, 'items'):
        return a.items()
    return enumerate(a)


class SkyPyDriver:
    r'''Class for running pipelines.

    This is the main class for running pipelines of functions with dependencies
    and using their results to generate variables and tables.
    '''

    def execute(self, configuration, file_format=None, overwrite=False):
        r'''Run a pipeline.

        This function runs a pipeline of functions to generate variables and
        the columns of a set of tables. It uses a Directed Acyclic Graph to
        determine a non-blocking order of execution that resolves any
        dependencies, see [1]_. Tables can optionally be written to file.

        Parameters
        ----------
        configuration : dict-like
            Configuration for the pipeline.
        file_format : str
            File format used to write tables. Files are written using the
            Astropy unified file read/write interface; see [2]_ for supported
            file formats. If None (default) tables are not written to file.
        overwrite : bool
            Whether to overwrite any existing files without warning.

        Notes
        -----
        Each step in the pipeline is configured by a dictionary specifying:

        - 'function' : the fully qualified name of the function
        - 'args' : a list of positional arguments (by value), or a dictionary
                   of keyword arguments

        Note that 'args' either specifices keyword arguments by value, or the
        names of previous steps in the pipeline and uses their return values as
        keyword arguments. Literal strings (i.e. not field names) are escaped
        by nested quotes: '"this is a literal string"' and "'another one'".

        'configuration' should contain the name and configuration of each
        variable and/or an entry named 'tables'. 'tables' should contain a set
        of nested dictionaries, first containing the name of each table, then
        the name and configuration of each column and optionally an entry named
        'init' with a configuration that initialises the table. If 'init' is
        not specificed the table will be initialised as an empty astropy Table
        by default.

        See [3]_ for examples of pipeline configurations in YAML format.

        References
        ----------
        .. [1] https://networkx.github.io/documentation/stable/
        .. [2] https://docs.astropy.org/en/stable/io/unified.html
        .. [3] https://github.com/skypyproject/skypy/tree/master/examples
        '''

        # config contains settings for all variables and table initialisation
        # table_config contains settings for all table columns
        config = deepcopy(configuration)
        table_config = config.pop('tables', {})
        default_table = {'function': 'astropy.table.Table'}
        config.update({k: v.pop('init', default_table)
                      for k, v in table_config.items()})

        # Create a Directed Acyclic Graph of all jobs and dependencies
        dag = networkx.DiGraph()

        # Variables initialised by value don't require function evaluations
        def isfunction(f):
            return isinstance(f, Mapping) and 'function' in f
        variables = {k: v for k, v in config.items() if not isfunction(v)}
        for v in variables:
            dag.add_node(v)
            setattr(self, v, config.pop(v))

        # Add nodes for each variable, table and column
        for job in config:
            dag.add_node(job)
        for table, columns in table_config.items():
            table_complete = '.'.join((table, 'complete'))
            dag.add_node(table_complete)
            for column in columns.keys():
                job = '.'.join((table, column))
                dag.add_node(job)

        # Add edges for all requirements and dependencies
        def deps(settings):
            deps = settings.get('depends', [])
            args = settings.get('args', [])
            for k, v in _items(args):
                if isinstance(v, str) and not (v[0] in '"\'' and v[0] == v[-1]):
                    deps.append(v)
            return deps
        for job, settings in config.items():
            dag.add_edges_from((d, job) for d in deps(settings))
        for table, columns in table_config.items():
            table_complete = '.'.join((table, 'complete'))
            dag.add_edge(table, table_complete)
            for column, settings in columns.items():
                job = '.'.join((table, column))
                dag.add_edge(table, job)
                dag.add_edge(job, table_complete)
                dag.add_edges_from((d, job) for d in deps(settings))

        # Execute jobs in order that resolves dependencies
        for job in networkx.topological_sort(dag):
            if job in variables or job.endswith('.complete'):
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
                getattr(self, table).write(filename, overwrite=overwrite)

    def _call_from_config(self, config):

        # Import function
        function_path = config.get('function').split('.')
        module = builtins
        for i, key in enumerate(function_path[:-1]):
            if not hasattr(module, key):
                module_name = '.'.join(function_path[:i+1])
                try:
                    module = import_module(module_name)
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(module_name)
            else:
                module = getattr(module, key)
        function = getattr(module, function_path[-1])

        # Parse arguments
        args = config.get('args', [])
        for k, v in _items(args):
            if isinstance(v, str):
                args[k] = self[v]

        # Call function
        if isinstance(args, Mapping):
            result = function(**args)
        else:
            result = function(*args)

        return result

    def __getitem__(self, label):
        # do not parse literal strings
        if label[0] in '"\'' and label[0] == label[-1]:
            return label[1:-1]

        name, key = re.search(r'^(\w*)\.?(\w*)$', label).groups()
        item = getattr(self, name)
        return item[key] if key else item
