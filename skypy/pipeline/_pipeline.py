"""Pipeline implementation.

This module provides methods to run pipelines of functions with dependencies
and handle their results.
"""

from astropy.cosmology import default_cosmology
from astropy.table import Table, Column
from copy import copy, deepcopy
from ._config import load_skypy_yaml
import networkx
import inspect


__all__ = [
    'Pipeline',
]


def infer(function, args, kwargs, context):
    '''infer missing function args and kwargs from context

    TODO: move this into the PipelineFunction() class once it exists
    '''

    try:
        # inspect the function
        sig = inspect.signature(function)
    except:
        # not all functions can be inspected
        sig = None

    if sig is not None:
        # inspect the function call for the given args and kwargs
        given = sig.bind_partial(*args, **kwargs)

        # now go through parameters one by one:
        # - check if the parameter has an argument given
        # - if not, check if the parameter has a default argument
        # - if not, check if the argument can be inferred from context
        for name, par in sig.parameters.items():
            if name in given.arguments:
                pass
            elif par.default is not par.empty:
                pass
            elif name in context:
                given.arguments[name] = context[name]

        # augment args and kwargs
        args.clear()
        args.extend(given.args)
        kwargs.clear()
        kwargs.update(given.kwargs)

    # return if successful
    return True if sig is not None else False


class Pipeline:
    r'''Class for running pipelines.

    This is the main class for running pipelines of functions with dependencies
    and using their results to generate variables and tables.
    '''

    @classmethod
    def read(cls, filename):
        '''Read a pipeline from a configuration file.

        Parameters
        ----------
        filename : str
            The name of the configuration file.

        '''
        config = load_skypy_yaml(filename)
        return cls(config)

    def __init__(self, configuration):
        '''Construct the pipeline.

        Parameters
        ----------
        configuration : dict-like
            Configuration for the pipeline.

        Notes
        -----
        Each step in the pipeline is configured by a dictionary specifying
        a variable name and the associated value.

        A value that is a tuple `(function, args, kwargs)` specifies that the
        value will be the result of a function call. The first item is a
        callable, and the second and third items specify the arguments.

        If a function argument is a string `$variable_name`, it refers to the
        values of previous step in the pipeline.

        'configuration' should contain the name and configuration of each
        variable and/or an entry named 'tables'. 'tables' should contain a set
        of nested dictionaries, first containing the name of each table, then
        the name and configuration of each column and optionally an entry named
        'init' with a configuration that initialises the table. If 'init' is
        not specificed the table will be initialised as an empty astropy Table
        by default.

        See [1]_ for examples of pipeline configurations in YAML format.

        References
        ----------
        .. [1] https://github.com/skypyproject/skypy/tree/master/examples

        '''

        # config contains settings for all variables and table initialisation
        # table_config contains settings for all table columns
        self.config = deepcopy(configuration)
        self.cosmology = self.config.pop('cosmology', None)
        self.parameters = self.config.pop('parameters', {})
        self.table_config = self.config.pop('tables', {})
        default_table = (Table, [], {})
        self.config.update({k: v.pop('.init', default_table)
                            for k, v in self.table_config.items()})

        # Initalise state with parameters
        self.state = copy(self.parameters)

        # Create a Directed Acyclic Graph of all jobs and dependencies
        self.dag = networkx.DiGraph()

        # context for function calls
        context = {}

        # use cosmology in global context if given
        if self.cosmology is not None:
            context['cosmology'] = '$cosmology'

        # - add nodes for each variable, table and column
        # - add edges for the table dependencies
        # - keep track where functions need to be called
        #   functions are tuples (function, args, kwargs)
        functions = {}
        for job, settings in self.config.items():
            self.dag.add_node(job, skip=False)
            if isinstance(settings, tuple):
                functions[job] = settings
                # infer additional function arguments from context
                function, args, kwargs = settings
                infer(function, args, kwargs, context)
        for table, columns in self.table_config.items():
            table_complete = '.'.join((table, 'complete'))
            self.dag.add_node(table_complete)
            self.dag.add_edge(table, table_complete)
            for column, settings in columns.items():
                job = '.'.join((table, column))
                self.dag.add_node(job, skip=False)
                self.dag.add_edge(table, job)
                self.dag.add_edge(job, table_complete)
                if isinstance(settings, tuple):
                    functions[job] = settings
                    # infer additional function arguments from context
                    function, args, kwargs = settings
                    infer(function, args, kwargs, context)
                # DAG nodes for individual columns in multi-column assignment
                names = [n.strip() for n in column.split(',')]
                if len(names) > 1:
                    for name in names:
                        subjob = '.'.join((table, name))
                        self.dag.add_node(subjob)
                        self.dag.add_edge(job, subjob)

        # go through functions and add edges for all references
        for job, settings in functions.items():
            # settings are tuple (function, args, kwargs)
            _, args, kwargs = settings
            # get dependencies from arguments
            deps = self.get_deps(args) + self.get_deps(kwargs)
            # add edges for dependencies
            for d in deps:
                # job depends on d
                self.dag.add_edge(d, job)
                # recurse dependencies such that d = 'a.b.c' -> 'a.b' -> 'a'
                c = d.rpartition('.')[0]
                while c:
                    self.dag.add_edge(c, d)
                    c, d = c.rpartition('.')[0], c

    def execute(self, parameters={}):
        r'''Run a pipeline.

        This function runs a pipeline of functions to generate variables and
        the columns of a set of tables. It uses a Directed Acyclic Graph to
        determine a non-blocking order of execution that resolves any
        dependencies, see [1]_.

        Parameters
        ----------
        parameters : dict
            Updated parameter values for this execution.

        References
        ----------
        .. [1] https://networkx.github.io/documentation/stable/

        '''
        # update parameter state
        self.parameters.update(parameters)

        # initialise state object
        self.state = copy(self.parameters)

        # Initialise cosmology from config parameters
        if self.cosmology is not None:
            self.state['cosmology'] = self.get_value(self.cosmology)

        # go through the jobs in dependency order
        for job in networkx.topological_sort(self.dag):
            node = self.dag.nodes[job]
            skip = node.get('skip', True)
            if skip:
                continue
            elif job in self.config:
                settings = self.config.get(job)
                self.state[job] = self.get_value(settings)
            else:
                table, column = job.split('.')
                settings = self.table_config[table][column]
                names = [n.strip() for n in column.split(',')]
                if len(names) > 1:
                    # Multi-column assignment
                    t = Table(self.get_value(settings), names=names)
                    self.state[table].add_columns(t.columns)
                else:
                    # Single column assignment
                    self.state[table][column] = self.get_value(settings)

    def write(self, file_format=None, overwrite=False):
        r'''Write pipeline results to disk.

        Parameters
        ----------
        file_format : str
            File format used to write tables. Files are written using the
            Astropy unified file read/write interface; see [1]_ for supported
            file formats. If None (default) tables are not written to file.
        overwrite : bool
            Whether to overwrite any existing files without warning.

        References
        ----------
        .. [1] https://docs.astropy.org/en/stable/io/unified.html

        '''
        if file_format:
            for table in self.table_config.keys():
                filename = '.'.join((table, file_format))
                self.state[table].write(filename, overwrite=overwrite)

    def get_value(self, value):
        '''return the value of a field

        tuples specify function calls `(function name, function args)`
        '''

        if isinstance(value, dict):
            # recurse dicts
            return {k: self.get_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            # recurse lists
            return [self.get_value(v) for v in value]
        elif isinstance(value, tuple):
            # tuple (function, args, kwargs)
            function, args, kwargs = value
            return function(*self.get_value(args), **self.get_value(kwargs))
        elif isinstance(value, str) and value[0] == '$':
            # reference
            return self[value[1:]]
        else:
            # plain value
            return value

    def get_deps(self, args):
        '''get dependencies from function args

        returns a list of all references found
        '''

        if isinstance(args, str) and args[0] == '$':
            # reference
            return [args[1:]]
        elif isinstance(args, tuple):
            # recurse on function arguments
            _, args, kwargs = args
            return self.get_deps(args) + self.get_deps(kwargs)
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
        item = self.state
        name = None
        while label:
            key, _, label = label.partition('.')
            name = f'{name}.{key}' if name else key
            try:
                item = item[key]
            except KeyError as e:
                raise KeyError('unknown label: ' + name) from e
        if isinstance(item, Column):
            return item.data if item.unit is None else item.quantity
        else:
            return item
