from scipy.stats import rv_discrete
from scipy._lib._util import check_random_state

import inspect


# list of exported symbols
__all__ = [
    'parametrise',
    'example_args',
]


# default examples
_EXAMPLES = """\
Examples
--------
>>> from %(module)s import %(name)s

Fix the example parameters:

>>> %(shapes)s = %(args)s

Calculate a few first moments:

>>> mean, var, skew, kurt = %(name)s.stats(%(shapes)s, moments='mvsk')

Check accuracy of ``cdf`` and ``ppf``:

>>> vals = %(name)s.ppf([0.001, 0.5, 0.999], %(shapes)s)
>>> np.allclose([0.001, 0.5, 0.999], %(name)s.cdf(vals, %(shapes)s))
True

Create a new plot:

>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 1)

Display the probability density function (``pdf``):

>>> x = np.linspace(%(name)s.ppf(0.01, %(shapes)s),
...                 %(name)s.ppf(0.99, %(shapes)s), 100)
>>> ax.plot(x, %(name)s.pdf(x, %(shapes)s),
...        'r-', lw=5, alpha=0.6, label='%(name)s pdf')

Generate random numbers:

>>> r = %(name)s.rvs(%(shapes)s, size=1000)

And compare the histogram:

>>> ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
>>> ax.legend(loc='best', frameon=False)
>>> plt.show()

"""


def parametrise(dist, argsfn, name=None, units=None):
    return rv_wrapped(dist, argsfn, name=name, units=units)


def example_args(*args):
    def decorator(f):
        f.example_args = args
        return f
    return decorator


class rv_wrapped(object):

    def _attach_units(self, x, power=1):
        if self.units is not None:
            if power != 1:
                return x*(self.units**power)
            else:
                return x*self.units
        return x

    def _detach_units(self, x):
        if self.units is not None:
            try:
                return x.to_value(self.units)
            except AttributeError:
                raise TypeError('units not a Quantity')
        return x

    def _get_shapes(self):
        sig = inspect.signature(self.argsfn)
        shapes = [p for p in sig.parameters]
        return ', '.join(shapes)

    def _make_doc(self, doc):
        # normalise indendation for substitution
        doc = inspect.cleandoc(doc)

        # collect example args into string
        args = ', '.join('%.3g' % arg for arg in self.example_args)

        # dictionary of substitutions
        docdict = {}
        docdict['module'] = self.module
        docdict['name'] = self.name
        docdict['shapes'] = self.shapes
        docdict['args'] = args
        docdict['examples'] = _EXAMPLES

        # recursive string substitution
        for i in range(5):
            _doc = doc % docdict
            if _doc == doc:
                break
            doc = _doc

        return doc

    def __init__(self, dist, argsfn, name=None, units=None, module=None,
                 example_args=None):
        # create a new rv instance
        self.dist = dist.__class__(**dist._updated_ctor_param())

        # store the args function
        self.argsfn = argsfn

        # get shape parameters of args function
        self.shapes = self._get_shapes()

        # set the name if given
        if name is not None:
            self.name = name
        else:
            # set to name of args function
            self.name = argsfn.__name__

        # set the module if given
        if module is not None:
            self.module = module
        else:
            # use module of args function
            self.module = argsfn.__module__

        # set the example args if given
        if example_args is not None:
            self.example_args = example_args
        elif hasattr(argsfn, 'example_args'):
            # example args given in args function
            self.example_args = argsfn.example_args
        else:
            # no example args
            self.example_args = ()

        # inherit docstring if not set
        if not self.__doc__:
            self.__doc__ = argsfn.__doc__

        # parse docstring
        self.__doc__ = self._make_doc(self.__doc__)

        # set units
        self.units = units

    @property
    def random_state(self):
        return self.dist._random_state

    @random_state.setter
    def random_state(self, seed):
        self.dist._random_state = check_random_state(seed)

    def pdf(self, x, *args, **kwargs):
        x = self._detach_units(x)
        args = self.argsfn(*args, **kwargs)
        return self.dist.pdf(x, *args)

    def logpdf(self, x, *args, **kwargs):
        x = self._detach_units(x)
        args = self.argsfn(*args, **kwargs)
        return self.dist.logpdf(x, *args)

    def cdf(self, x, *args, **kwargs):
        x = self._detach_units(x)
        args = self.argsfn(*args, **kwargs)
        return self.dist.cdf(x, *args)

    def logcdf(self, x, *args, **kwargs):
        x = self._detach_units(x)
        args = self.argsfn(*args, **kwargs)
        return self.dist.logcdf(x, *args)

    def ppf(self, q, *args, **kwargs):
        args = self.argsfn(*args, **kwargs)
        return self._attach_units(self.dist.ppf(q, *args))

    def isf(self, q, *args, **kwargs):
        args = self.argsfn(*args, **kwargs)
        return self._attach_units(self.dist.isf(q, *args))

    def rvs(self, *args, **kwargs):
        size = kwargs.pop('size', None)
        rndm = kwargs.pop('random_state', None)
        args = self.argsfn(*args, **kwargs)
        rvs = self.dist.rvs(*args, size=size, random_state=rndm)
        return self._attach_units(rvs)

    def sf(self, x, *args, **kwargs):
        x = self._detach_units(x)
        args = self.argsfn(*args, **kwargs)
        return self.dist.sf(x, *args)

    def logsf(self, x, *args, **kwargs):
        x = self._detach_units(x)
        args = self.argsfn(*args, **kwargs)
        return self.dist.logsf(x, *args)

    def stats(self, *args, **kwargs):
        moments = kwargs.pop('moments', 'mv')
        args = self.argsfn(*args, **kwargs)
        return self.dist.stats(*args, moments=moments)

    def median(self, *args, **kwargs):
        args = self.argsfn(*args, **kwargs)
        return self._attach_units(self.dist.median(*args))

    def mean(self, *args, **kwargs):
        args = self.argsfn(*args, **kwargs)
        return self._attach_units(self.dist.mean(*args))

    def var(self, *args, **kwargs):
        args = self.argsfn(*args, **kwargs)
        return self._attach_units(self.dist.var(*args), 2)

    def std(self, *args, **kwargs):
        args = self.argsfn(*args, **kwargs)
        return self._attach_units(self.dist.std(*args))

    def moment(self, n, *args, **kwargs):
        args = self.argsfn(*args, **kwargs)
        return self.dist.moment(n, *args)

    def entropy(self, *args, **kwargs):
        args = self.argsfn(*args, **kwargs)
        return self.dist.entropy(*args)

    def pmf(self, k, *args, **kwargs):
        args = self.argsfn(*args, **kwargs)
        return self.dist.pmf(k, *args)

    def logpmf(self, k, *args, **kwargs):
        args = self.argsfn(*args, **kwargs)
        return self.dist.logpmf(k, *args)

    def interval(self, alpha, *args, **kwargs):
        args = self.argsfn(*args, **kwargs)
        return self.dist.interval(alpha, *args)

    def expect(self, func=None, args=(), lb=None, ub=None, conditional=False,
               **kwds):
        args = self.argsfn(*args)
        a, loc, scale = self.dist._parse_args(*args)
        if isinstance(self.dist, rv_discrete):
            return self.dist.expect(func, a, loc, lb, ub, conditional, **kwds)
        else:
            return self.dist.expect(func, a, loc, scale, lb, ub,
                                    conditional, **kwds)

    def support(self, *args, **kwargs):
        args = self.argsfn(*args, **kwargs)
        return self.dist.support(*args)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('freezing not yet implemented')
