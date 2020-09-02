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

        If a function argument is a tuple `(variable_name,)`, it refers to the
        values of previous step in the pipeline. The tuple item must be the
        name of the reference variable.

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
            # get dependencies from arguments
            deps = self.get_deps(args)
            # add edges for dependencies
            for d in deps:
                if dag.has_node(d):
                    dag.add_edge(d, job)
                else:
                    raise KeyError(d)

        # Execute jobs in order that resolves dependencies
        for job in networkx.topological_sort(dag):
            if job.endswith('.complete'):
                continue
            elif job in config:
                settings = config.get(job)
                setattr(self, job, self.get_value(settings))
            else:
                table, column = job.split('.')
                settings = table_config[table][column]
                getattr(self, table)[column] = self.get_value(settings)

        # Write tables to file
        if file_format:
            for table in table_config.keys():
                filename = '.'.join((table, file_format))
                getattr(self, table).write(filename, overwrite=overwrite)

    def get_value(self, value):
        '''return the value of a field

        tuples specify function calls `(function name, function args)`
        '''

        # check for plain value
        if not isinstance(value, tuple):
            return value

        # value is tuple (function name, [function args])
        name = value[0]
        args = value[1] if len(value) > 1 else []

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

        # Parse arguments
        parsed_args = self.get_args(args)

        # Call function
        if isinstance(args, dict):
            result = function(**parsed_args)
        elif isinstance(args, list):
            result = function(*parsed_args)
        else:
            result = function(parsed_args)

        return result

    def get_args(self, args):
        '''parse function arguments

        tuples specify references to container: `(field name,)`
        '''

        if isinstance(args, tuple):
            # get reference
            return self[args[0]]
        elif isinstance(args, dict):
            # recurse kwargs
            return {k: self.get_args(v) for k, v in args.items()}
        elif isinstance(args, list):
            # recurse args
            return [self.get_args(a) for a in args]
        else:
            # return value
            return args

    def get_deps(self, args):
        '''get dependencies from function args

        returns a list of all dependencies found
        '''

        if isinstance(args, tuple):
            # reference
            return [args[0]]
        elif isinstance(args, dict):
            # get explicit dependencies
            deps = args.pop('.depends', [])
            # turn a single value into a list
            if isinstance(deps, str) or not isinstance(deps, list):
                deps = [deps]
            # recurse remaining kwargs
            return deps + sum([self.get_deps(a) for a in args.values()], [])
        elif isinstance(args, list):
            # recurse args
            return sum([self.get_deps(a) for a in args], [])
        else:
            # no reference
            return []

    def __getitem__(self, label):
        name, _, key = label.partition('.')
        item = getattr(self, name)
        return item[key] if key else item
