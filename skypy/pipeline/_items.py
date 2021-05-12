'''item types in the pipeline'''

from collections.abc import Sequence, Mapping
from . import log
import inspect


class Item:
    '''base class for items in the pipeline'''

    def infer(self, context):
        '''infer additional properties from context'''
        pass

    def depend(self, pipeline):
        '''return list of dependencies'''
        return []

    def evaluate(self, pipeline):
        '''return computed value of item'''
        return None


class Ref(Item):
    '''reference to another item'''

    def __init__(self, ref):
        self.ref = ref

    def depend(self, pipeline):
        return [self.ref]

    def evaluate(self, pipeline):
        return pipeline[self.ref]


class Call(Item):
    '''item that calls a function'''

    def __init__(self, function, args=[], kwargs={}):
        '''initialise the call'''

        if not callable(function):
            raise TypeError('function is not callable')
        if not isinstance(args, Sequence):
            raise TypeError('args is not a sequence')
        if not isinstance(kwargs, Mapping):
            raise TypeError('kwargs is not a mapping')

        self.function = function
        self.args = args
        self.kwargs = kwargs

    def infer(self, context):
        '''infer missing function args and kwargs from context'''

        try:
            # inspect the function
            sig = inspect.signature(self.function)
        except ValueError:
            # not all functions can be inspected
            sig = None

        if sig is not None:
            # inspect the function call for the given args and kwargs
            given = sig.bind_partial(*self.args, **self.kwargs)

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
            self.args = given.args
            self.kwargs = given.kwargs

    def depend(self, pipeline):
        '''return a list of dependencies of the call'''
        return pipeline.depend(self.args) + pipeline.depend(self.kwargs)

    def evaluate(self, pipeline):
        '''execute the call in the given pipeline'''
        args = pipeline.evaluate(self.args)
        kwargs = pipeline.evaluate(self.kwargs)
        log.info(f"Calling {self.function.__name__}")
        return self.function(*args, **kwargs)
