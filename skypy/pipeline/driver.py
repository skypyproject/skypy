"""Driver module.

This module provides methods to run pipelines of functions with dependencies
and handle their results.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib import import_module
import builtins
import networkx


__all__ = [
    'SkyPyDriver',
    'LiteralValue',
]


def _items(a):
    '''return keys and values for dict or list'''
    if hasattr(a, 'items'):
        return a.items()
    return enumerate(a)


class LiteralValue:
    '''Tag that marks fields as literal variables.'''

    def __init__(self, value):
        self.value = value


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
        default_table = {'astropy.table.Table': []}
        config.update({k: v.pop('init', default_table)
                      for k, v in table_config.items()})

        # Create a Directed Acyclic Graph of all jobs and dependencies
        dag = networkx.DiGraph()

        # Add nodes for each variable, table and column
        for job in config:
            dag.add_node(job)
        for table, columns in table_config.items():
            table_complete = '.'.join((table, 'complete'))
            dag.add_node(table_complete)
            for column in columns.keys():
                job = '.'.join((table, column))
                dag.add_node(job)

        # this function returns the dependencies given a field
        def deps(field):
            # strings specify dependencies
            if isinstance(field, str):
                return [field]
            # dicts specify function calls
            if isinstance(field, Mapping):
                d = field.pop('depends', [])
                for args in field.values():
                    for k, v in _items(args):
                        # string arguments are dependencies
                        if isinstance(v, str):
                            d.append(v)
                return d
            # recurse over lists
            if isinstance(field, Sequence):
                return sum([deps(item) for item in field], [])
            # everything else has no dependencies
            return []

        # Add edges for all requirements and dependencies
        for job, settings in config.items():
            for d in deps(settings):
                if not dag.has_node(d):
                    raise KeyError(d)
                dag.add_edge(d, job)
        for table, columns in table_config.items():
            table_complete = '.'.join((table, 'complete'))
            dag.add_edge(table, table_complete)
            for column, settings in columns.items():
                job = '.'.join((table, column))
                dag.add_edge(table, job)
                dag.add_edge(job, table_complete)
                for d in deps(settings):
                    if not dag.has_node(d):
                        raise KeyError(d)
                    dag.add_edge(d, job)

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
                getattr(self, table).write(filename, overwrite=overwrite)

    def _call_from_config(self, field):
        # handle explicit variables
        if isinstance(field, LiteralValue):
            return field.value

        # handle references == strings
        if isinstance(field, str):
            return self[field]

        # handle lists by recursion
        if isinstance(field, Sequence):
            return sum([self._call_from_config(item) for item in field], [])

        # handle functions == dicts
        if isinstance(field, Mapping):

            # calls only the first function in dict really
            for function, args in field.items():

                # import function
                function_path = function.split('.')
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
                for k, v in _items(args):
                    if isinstance(v, str):
                        # strings are references
                        args[k] = self[v]
                    elif isinstance(v, LiteralValue):
                        # literal variable
                        args[k] = v.value

                # Call function
                if isinstance(args, Mapping):
                    result = function(**args)
                else:
                    result = function(*args)

                return result

        # handle everything else by returning it
        return field

    def __getitem__(self, label):
        name, _, key = label.partition('.')
        item = getattr(self, name)
        return item[key] if key else item
