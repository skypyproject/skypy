"""Driver module.

This module provides methods to run pipelines of functions with dependencies
and handle their results.
"""

from copy import deepcopy
from importlib import import_module
import builtins
import networkx


__all__ = [
    'SkyPyDriver',
]


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
        Each step in the pipeline is configured by a dictionary specifying
        a variable name and the associated value.

        A value that is a tuple `(function_name, function_args)` specifies that
        the value will be the result of a function call. The first item is the
        fully qualified function name, and the second value specifies the
        function arguments.

        If a function argument is a tuple `(variable_name, default_value)`, it
        refers to the values of previous step in the pipeline. The first item
        is the name of the reference variable, and the optional second argument
        is a default value in case the variable cannot be found.

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
        default_table = ('astropy.table.Table',)
        config.update({k: v.pop('.init', default_table)
                      for k, v in table_config.items()})

        # Create a Directed Acyclic Graph of all jobs and dependencies
        dag = networkx.DiGraph()

        # - add nodes for each variable, table and column
        # - add edges for the table dependencies
        # - keep track where functions need to be called
        #   functions are tuples (function name, [function args])
        functions = {}
        for job, settings in config.items():
            dag.add_node(job)
            if isinstance(settings, tuple):
                functions[job] = settings
        for table, columns in table_config.items():
            table_complete = '.'.join((table, 'complete'))
            dag.add_node(table_complete)
            dag.add_edge(table, table_complete)
            for column, settings in columns.items():
                job = '.'.join((table, column))
                dag.add_node(job)
                dag.add_edge(table, job)
                dag.add_edge(job, table_complete)
                if isinstance(settings, tuple):
                    functions[job] = settings

        # go through functions and add edges for all references
        for job, settings in functions.items():
            # settings are tuple (function, [args])
            args = settings[1] if len(settings) > 1 else None
            # three kinds of arguments: kwargs, args, values
            if isinstance(args, dict):
                deps = args.pop('.depends', [])
                # turn a single values into a list
                if isinstance(deps, str) or not isinstance(deps, list):
                    deps = [deps]
                # make all explicit deps into tuples
                deps = [(d,) for d in deps]
                # get dependencies from mapping
                deps += [a for a in args.values() if isinstance(a, tuple)]
            elif isinstance(args, list):
                # get dependencies from sequence
                deps = [a for a in args if isinstance(a, tuple)]
            else:
                # get single dependency
                deps = [args] if isinstance(args, tuple) else []
            # add edges for dependencies
            for d in deps:
                if dag.has_node(d[0]):
                    dag.add_edge(d[0], job)
                elif len(d) < 2:
                    # no default argument
                    raise KeyError(d[0])

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

    def _call_from_config(self, config):

        # check for variable
        if not isinstance(config, tuple):
            return config

        # config is tuple (function name, [function args])
        name = config[0]
        args = config[1] if len(config) > 1 else []

        # Import function
        function_path = name.split('.')
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

        # parse arguments and call function
        if isinstance(args, (dict, list)):
            # kwargs or args
            for k, v in args.items() if isinstance(args, dict) else enumerate(args):
                if isinstance(v, tuple):
                    args[k] = self.get(*v)
            result = function(**args) if isinstance(args, dict) else function(*args)
        else:
            # value
            if isinstance(args, tuple):
                args = self.get(*args)
            result = function(args)

        return result

    def get(self, label, default=None):
        name, _, key = label.partition('.')
        try:
            item = getattr(self, name)
            return item[key] if key else item
        except (KeyError, AttributeError):
            return default
