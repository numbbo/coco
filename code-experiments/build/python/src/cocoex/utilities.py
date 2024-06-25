import ast as _ast  # avoid name space polluting
import os as _os
import sys as _sys
import time as _time
import warnings as _warnings
import collections as _collections  # defaultdict, OrderedDict

import numpy as np


def about_equal(a, b, precision=1e-6):
    """
    Return True if the floating point number ${a} and ${b} are about equal.
    """
    if a == b:
        return True

    absolute_error = abs(a - b)
    larger = a if abs(a) > abs(b) else b
    relative_error = abs(a - b) / (abs(larger) + 2 * _sys.float_info.min)

    if absolute_error < (2 * _sys.float_info.min):
        return True
    return relative_error < precision


def args_to_dict(args, known_names, specials=None, split='=',
                 print=lambda *args, **kwargs: None):
    """return a `dict` from a list of ``"name=value"`` strings.

    `args` come in the form of a `list` of ``"name=value"`` strings
    without spaces, like ``["budget_multiplier=100"]``.

    Return ``dict(arg.split(split) for arg in args)`` in the most
    basic case, but additionally (i) checks that the keys of this
    `dict` are known names, (ii) evaluates the values in some cases
    and (iii) handles `specials`.

    `known_names` is an iterable (`dict` or `list` or `tuple`) of strings.
    If ``known_names is None``, all args are processed, otherwise a
    `ValueError` is raised for unknown names. This is useful if we
    want to re-assign variables (overwrite default values) and avoid
    spelling mistakes pass silently.

    The value is processed as a Python literal with `ast.literal_eval`
    or remains a `str` when this is unsuccessful.

    `specials` is a `dict` and can currently only contain ``'batch'``,
    followed by ``"name1/name2"`` as value. ``name1`` and ``name2`` are
    then assigned from the values in `arg`, for example to 2 and 4 with
    ``batch=2/4``.

    A main usecase is to process ``sys.argv[1:]`` into a `dict` in a
    python script, like::

        command_line_dict = args_to_dict(sys.argv[1:], globals())
        globals().update(command_line_dict)

    >>> import cocoex
    >>> d = cocoex.utilities.args_to_dict(["budget=2.3", "bed=bed-name", "number=4"],
    ...                                   ["budget", "bed", "number", "whatever"])
    >>> len(d)
    3
    >>> assert d['bed'] == 'bed-name'
    >>> assert isinstance(d["budget"], float)

    """
    def eval_value(value):
        try:
            return _ast.literal_eval(value)
        except Exception:  # ValueError or SyntaxError or ??
            return value
    res = {}
    for arg in args:
        name, value = arg.split(split)
        # what remains to be done is to verify name,
        # compute non-string value, and assign res[name] = value
        if specials and name in specials:
            if name == 'batch':
                print('batch:')
                if len(specials['batch'].split('/')) != len(value.split('/')):
                    raise ValueError("'{0}' is not a valid value for argument 'batch={0}'."
                                     " A valid example is 'batch=1/3' ({1})"
                                     .format(value, specials['batch']))  # zip(..., strict=...) bails in Python3.9
                for k, v in zip(specials['batch'].split('/'), value.split('/')):
                    res[k] = int(v)  # batch accepts only int
                    print(' ', k, '=', res[k])
                continue  # name is processed
            else:
                raise ValueError(name, 'is unknown special')
        for known_name in known_names if known_names is not None else [name]:
            # check that name is an abbreviation of known_name and unique in known_names
            if known_name.startswith(name) and (
                        sum([other.startswith(name)
                             for other in known_names or [name]]) == 1):
                if known_name in res:
                    if res[known_name] != eval_value(value):
                        raise ValueError("found two values for argument {0}={1} and ={2}"
                                         .format(known_name, res[known_name], value))
                else:
                    res[known_name] = eval_value(value)
                print(known_name, '=', res[known_name])
                break  # name == arg.split()[0] is processed
        else:
            raise ValueError('Argument name "{}" is ambiguous or not given in ``known_names=={}``'
                             .format(name, sorted([k for k in known_names if not k.startswith('_')
                                    and k not in ('division', 'print_function', 'unicode_literals',
                                                  'sys', 'warnings', 'time', 'defaultdict', 'os', 'np', 'scipy')])))
    return res

def dict_to_eval(dict_, ignore_list=('_', 'self')):
    """return a pruned `dict` so that ``ast.literal_eval(repr(dict_))`` works.

    Keys that start with entries from `ignore_list` when interpreted as
    `str` are removed, by default those starting with ``_`` or ``self``,
    where `ignore_list` must be a `str` or a `tuple` of `str`.

    See also `write_setting`.
    """
    res = {}
    for k in dict_:
        if str(k).startswith(ignore_list):
            continue
        try:
            _ast.literal_eval(repr(dict_[k]))
        except Exception:
            pass
        else:
            res[k] = dict_[k]
    return res

def write_setting(dict_, filename, ignore_list=None):
    """write a simplified parameters dictionary to a file,

    for keeping a record or, e.g., for checking the `budget_multiplier`
    later.

    A typical usecase is ``write_setting(locals(), 'parameters.pydat')``.

    When ``ignore_list is not None`` it is passed to `dict_to_eval`
    which determines which parameters are written or omitted. By default,
    keys starting with ``_`` or ``self`` are omitted and items that bail on
    `literal_eval`.

    See also `dict_to_eval` and `read_setting`.
    """
    if isinstance(filename, (tuple, list)):
        filename = _os.path.join(*filename)
    if not _os.path.exists(filename):
        with open(filename, 'wt') as f:
            if ignore_list is None:
                f.write(repr(dict_to_eval(dict_)))
            else:
                f.write(repr(dict_to_eval(dict_, ignore_list)))
    else:
        _warnings.warn('nothing written as the file "' + filename +
                       '"\nexists already (this should never happen as\n'
                       'each experiment should be written in a new (sub-)folder)')

def read_setting(filename, warn=True):
    """return file content evaluated as Python literal (e.g. a `dict`),

    return `None` if `filename` is not a valid path.

    If `warn`, throw a warning when the file `filename` does not exist.

    A typical usecase could be
    ``old_multiplier = read_setting('parameters.pydat')['budget_multiplier']``.

    See also `write_setting`.
    """
    if _os.path.exists(filename):
        with open(filename, 'rt') as f:
            return _ast.literal_eval(f.read())
    warn and _warnings.warn("Parameter file '{}' (to check setting consistency)"
                            " does not exist".format(filename))

def forgiving_import(module, warn_level=0):
    """Do nothing if import fails, return the imported module otherwise.

    Usage::

        cma = forgiving_import('cma')

    in place of::

        import cma

    This is helpful to keep some code smoothly working when `cma` is not
    installed and not used in the current use case.
    """
    try:
        import importlib
    except ImportError:
        print('Please replace \n\n   {}\n\n with \n\n    {}\n\n'
              'or incomment the respective line.'.format(
            "module = forgiving_import('module')", 
            "import module"))
        raise
    try:
        return importlib.import_module(module)
    except ImportError:
        if warn_level:
            print("importing module `{}` with `importlib` failed".format(module))
        if warn_level > 1:
            raise

class ObserverOptions(dict):
    """a `dict` with observer options which can be passed to
    the (C-based) `Observer` via the `as_string` property.

    See http://numbbo.github.io/coco-doc/C/#observer-parameters
    for details on the available (C-based) options.

    Details: When the `Observer` class in future accepts a dictionary
    also, this class becomes superfluous and could be replaced by a method
    `default_observer_options` similar to `default_observers`.
    """
    def __init__(self, options={}):
        """set default options from global variables and input ``options``.

        Default values are created "dynamically" based on the setting
        of module-wide variables `SOLVER`, `suite_name`, and `budget`.
        """
        dict.__init__(self, options)
    def update(self, *args, **kwargs):
        """add or update options"""
        dict.update(self, *args, **kwargs)
        return self
    def update_gracefully(self, options):
        """update from each entry of parameter ``options: dict`` but only
        if key is not already present
        """
        for key in options:
            if key not in self:
                self[key] = options[key]
        return self
    @property
    def as_string(self):
        """string representation which is accepted by `Observer` class,
        which calls the underlying C interface
        """
        s = str(self).replace(',', ' ')
        for c in ["u'", "'", "{", "}"]:
            s = s.replace(c, '')
        return s


class ProblemNonAnytime:
    """The non-anytime problem class.

    Serves to benchmark a "budgeted" algorithm whose behavior decisively
    depends on a budget input parameter.

    Usage::

        # given: suite and observer instances, budget_list, fmin
        for index in range(len(suite)):
            with ProblemNonAnytime(suite, observer, index) as problem:
                x0 = problem.initial_solution  # or whatever fmin needs as input
                for budget in sorted(budget_list):
                    x = fmin(problem, x0, budget)  # minimize
                    problem.delivered(x)  # prepares for next budget also
                    if problem.final_target_hit:
                        break

    Details: This class maintains two problems - one observed and the
    other unobserved. It switches from unobserved to observed when
    ``p_unobserved.evaluations >= p_observed.evaluations``.

    """
    inherited_constant_attributes = [
            'dimension',
            'id',
            'index',
            'info',
            'initial_solution',
            'lower_bounds',
            'name',
            'number_of_constraints',
            'number_of_objectives',
            'number_of_variables',
            'upper_bounds']

    def __init__(self, suite, observer, index):
        self.suite = suite
        self.p_unobserved = suite[index]
        self.p_observed = suite[index].add_observer(observer)
        self._p = self.p_observed
        self.evaluations = 0
        self._number_of_delivery_evaluations = 0  # just FTR
        for key in ProblemNonAnytime.inherited_constant_attributes:
            setattr(self, key, getattr(self._p, key))

    def __call__(self, x, *args, **kwargs):
        if self.evaluations > self.p_observed.evaluations:
            raise ValueError(
                "Evaluations {0} are larger than observed evaluations {1}"
                "".format(self.evaluations, self.p_observed.evaluations))
        if self.evaluations >= self.p_observed.evaluations:
            self._p = self.p_observed
        self.evaluations += 1
        return self._p(x, *args, **kwargs)

    def delivered(self, x, *args, **kwargs):
        """to be called with the solution returned by the solver.

        The method records the delivered solution (if necessary) and
        prepares the problem to be run on the next budget by calling
        `reset`.
        """
        if self.evaluations < self.p_observed.evaluations:
            raise ValueError(
                "Delivered solutions should come from increasing budgets,\n"
                "however the current budget = {0} is smaller than the "
                "delivery budget = {1}".format(
                    self.p_observed.evaluations, self.evaluations))
        # assuming that "same fitness" means the solution was already evaluated
        if self.p_unobserved(x, *args, **kwargs) != self.best_observed_fvalue1:
            # ideally we would decrease the observed evaluation counter,
            # but instead we now increase it only one time step later in
            # __call__. That is, in effect, we exchange the next solution
            # to be observed with this delivered solution `x`.
            self.p_observed(x, *args, **kwargs)
            self._number_of_delivery_evaluations += 1  # just FTR
        self.reset()

    def reset(self):
        """prepare the problem to be run on the next budget"""
        self.evaluations = 0
        self._p = self.p_unobserved

    def free(self):
        self.p_observed.free()
        self.p_unobserved.free()

    def print_progress(self):
        if not self.index % (len(self.suite) / len(self.suite.dimensions)):
            print(f"\nd={self.dimension}: ")
        print("{}{} ".format(
            self.id[self.id.index("_f") + 1:self.id.index("_i")],
            self.id[self.id.index("_i"):self.id.index("_d")]), end="",
              flush=True)
        if not (self.index + 1) % 10:
            print("")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()

    @property
    def final_target_hit(self):
        return self.p_observed.final_target_hit

    @property
    def best_observed_fvalue1(self):
        return self.p_observed.best_observed_fvalue1

    @property
    def random_solution(self):
        return self.lower_bounds + (
            (np.random.rand(self.dimension) + np.random.rand(self.dimension))
            * (self.upper_bounds - self.lower_bounds) / 2)


class SameFunction:
    """Count the number of consecutive instances of the same function.

    Useful to limit the number of repetitions on the same function.

    Example:

    >>> import cocoex
    >>> from cocoex.utilities import SameFunction
    >>>
    >>> suite = cocoex.Suite('bbob', '', '')
    >>> already_seen = SameFunction()
    >>> processed = 0
    >>> for problem in suite:
    ...     if already_seen(problem.id) > 5:
    ...         continue
    ...     # do something here only with the first five instances
    ...     processed += 1
    >>> processed, len(suite), already_seen.count
    (864, 2160, 15)

    More arbitrary tests:

    >>> for i in range(4):
    ...     if seen('f001_i%d' % i) > 2:
    ...         continue
    ...     # do something here only the first two instances
    >>> seen.count
    4
    >>> seen('f_d03_i001')
    0
    >>> seen('f_d03_i02')
    1
    >>> for i in range(4):
    ...     if seen('f%d_i%d' % (i, i)):
    ...         break
    >>> i, seen.count
    (3, 1)

    """
    @staticmethod
    def filter(id):
        """remove instance information and return a `tuple`"""
        return tuple(i for i in id.split('_') if not i.startswith('i'))
    def __init__(self):
        self.count = 0
    def __call__(self, id):
        """return number of directly preceding calls with similar `id`
        """
        new = SameFunction.filter(id)
        if self.count == 0 or new != self.last:
            self.last = new
            self.count = 0
        self.count += 1
        return self.count - 1  # return old count


class MiniPrint:
    """print dimension when changed and a single symbol for each call.

    Details: print '|' if ``problem.final_target_hit``, ':' if restarted
    and '.' otherwise.
    """
    def __init__(self):
        self.dimension = None
        self.id_function = None
        self.last_index = -1
        self._calls = 0  # formatting aid
        self._functions = 0  # not in use
        self._sweeps = 0
        self._day0 = _time.localtime()[2]
    @property
    def stime(self):
        """current time as string +days since started"""
        ltime = _time.localtime()[2:6]
        s = "%dh%02d:%02d" % tuple(ltime[1:])
        if ltime[0] > self._day0:
            s = s + "+%dd" % (ltime[0] - self._day0)
        return s
    def __call__(self, problem, final=False, restarted=False, final_message=False):
        new_dimension = self.dimension != problem.dimension or (
            self.last_index > problem.index)
        if new_dimension:
            _sys.stdout.flush()
            if self.dimension is not None:
                print('')
            print("%dD %s" % (problem.dimension, self.stime))
            self.dimension = problem.dimension
            self._calls = 0
            self._functions = 0
        # elif not self._calls % 10:
        #     if self._calls % 50:
        #         print(' ', end='')
        #     else:
        #         print()
        if self.id_function != problem.id_function or new_dimension:
            self.id_function = problem.id_function
            if not new_dimension and self._calls >= 50:
                self._calls = 0
                print()
            print('{}f{}'.format(' ' if self.id_function < 10 else '',
                                 self.id_function), end='', flush=True)
            self._functions += 1
        self.last_index = problem.index
        self._calls += 1
        print('|' if problem.final_target_hit else ':' if restarted else '.', end='')
        if final:  # final print
            self._sweeps += 1
            if final_message:
                try:
                    len(final_message)  # poor mans string type check
                    print(final_message)
                except TypeError:
                    print('\nSuite done ({})'.format(self._sweeps))
        _sys.stdout.flush()


class ShortInfo:
    """print minimal info during benchmarking.

    After initialization, to be called right before the solver is called with
    the respective problem. Prints nothing if only the instance id changed.

    Example output:

        Jan20 18h27:56, d=2, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

        Jan20 18h27:56, d=3, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

        Jan20 18h27:57, d=5, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

    """
    def __init__(self):
        self.f_current = None  # function id (not problem id)
        self.d_current = 0  # dimension
        self.t0_dimension = _time.time()
        self.evals_dimension = 0
        self.evals_by_dimension = {}
        self.runs_function = 0

    def print(self, problem, end="", **kwargs):
        print(self(problem), end=end, **kwargs)
        _sys.stdout.flush()

    def add_evals(self, evals, runs):
        self.evals_dimension += evals
        self.runs_function += runs

    def dimension_done(self):
        self.evals_by_dimension[self.d_current] = (_time.time() - self.t0_dimension) / self.evals_dimension
        s = '\n    %d-D done in %.1e seconds/evaluation' % (self.d_current, self.evals_by_dimension[self.d_current])
        # print(self.evals_dimension)
        self.evals_dimension = 0
        self.t0_dimension = _time.time()
        return s

    def function_done(self):
        s = "(%d)" % self.runs_function + (2 - int(np.log10(self.runs_function))) * ' '
        self.runs_function = 0
        return s

    def __call__(self, problem):
        """uses `problem.id` and `problem.dimension` to decide what to print.
        """
        f = "f" + problem.id.lower().split('_f')[1].split('_')[0]
        res = ""
        ran_once = self.f_current is not None
        f_changed = f != self.f_current
        d_changed = problem.dimension != self.d_current
        run_complete = f_changed or d_changed
        if ran_once and run_complete:
            res += self.function_done() + ' '
        if d_changed:
            res += '%s%s, d=%d, running: ' % (
                        self.dimension_done() + "\n\n" if self.d_current else '',
                        ShortInfo.short_time_stap(),
                        problem.dimension)
            self.d_current = problem.dimension
        if run_complete:
            res += '%s' % f
        if f_changed:
            self.f_current = f
        # print_flush(res)
        return res

    def print_timings(self):
        print("  dimension seconds/evaluations")
        print("  -----------------------------")
        for dim in sorted(self.evals_by_dimension):
            print("    %3d      %.1e " %
                  (dim, self.evals_by_dimension[dim]))
        print("  -----------------------------")

    @staticmethod
    def short_time_stap():
        t = _time.asctime().split()
        d = t[0]
        d = t[1] + t[2]
        h, m, s = t[3].split(':')
        return d + ' ' + h + 'h' + m + ':' + s


def ascetime(sec, decimals=0):
    """return elapsed time as str.

    Example: return ``"0h33:21" if sec == 33*60 + 21``.
    """
    if sec < 0:  # sort-of ValueError?
        return "%.1f sec" % (sec)
    ds = str(sec % 1)[1:2+decimals] if decimals else ""  # fraction
    # if sec < 10:  # breaks backwards compatibility of processing printed output
    #     return "%d%s sec" % (sec, str(sec % 1)[1:2+max((1, decimals))])
    if sec < 100 and decimals:
        return "%d%s sec" % (sec, ds)
    h = sec / 60**2  # hours
    m = 60 * (h % 1)  # minutes
    s = 60 * (m % 1)  # seconds
    return "%dh%02d:%02d%s" % (h, m, s, ds)


def print_flush(*args):
    """print without newline but with flush"""
    print(*args, end="")
    _sys.stdout.flush()

class ExperimentRepeater:
    """Allow to automatically repeat an experiment based on budget and successes.

    The class tracks evaluations and the `final_target_hit` status
    ("successes") of the problem instances of a `cocoex.Suite`. Based on
    the stored information, the `done` method allows to run sub-experiment
    sweeps until some given budget is exhaused, namely, ``budget_multiplier
    * dimension``.

    Example code snippet::

        [...]
        repeater = cocoex.ExperimentRepeater(budget_multiplier)
        [...]
        while not repeater.done():
            for problem in suite:
                if repeater.done(problem):
                    continue
                problem.observe_with(observer)  # generates the data for cocopp postprocessing
                fmin(problem, repeater.initial_solution_proposal(problem))
                repeater.track(problem)  # record evaluations and success

    The call ``repeater.done()`` checks whether the budget is exhausted or
    enough trials were successful on all problems, or an instance lacks a
    trial on some problems, it is akin to, but not exactly the same as,
    ``all(repeater.done(p) for p in suite)``. ``ExperimentRepeater(0)``
    does exactly one sweep and no repetitions.

    The stored data can be queried, in which case `problem` can be a
    ``(id_function, dimension, id_instance)`` tuple. In particular, the
    methods `evaluations`, `successes` and `trials` return a `list` of the
    respective data for each instance of `instances`. The last entry of the
    argument tuple is ignored in these calls and can be `None` or omitted.
    The `all` method allows to query all `(id_function, dimension)` entries
    at once::

        >> evals = repeater.all(repeater.evaluations)  # returns a `dict`
        >> fun, dim = 1, 10
        >> evals[(fun, dim)] == repeater.evaluations((fun, dim, None))
        True

    Terminology: a "problem instance" is defined by the triple
    ``(id_function, dimension, id_instance)`` and "problem" may refer
    to any or all problem instances with the same ``(id_function,
    dimension)``.

    Details
    -------
    Only the `track` method relies on passing a `cocoex.Problem`
    instance, otherwise a ``(id_function, dimension, id_instance)``
    tuple is eligible. The tracked information is stored in the `._data`
    dictionary with ``(id_function, dimension)`` as keys. The `data` method
    implements safe access to this dictionary.

    When problem instances are repeated in a single suite, they may be
    _partially_ skipped _after_ the first full sweep. That is, the
    configuration ``1-5,1-5,1-5`` can also lead to four trials of each
    instance 1-5, because all instances have been repeated the same number
    of times.

    >>> import cocoex  # some not very meaningful testing
    >>> repeater = cocoex.ExperimentRepeater(2)
    >>> assert not repeater.done((1, 2))
    >>> assert repeater._sweeps == 0 and not repeater.done() and repeater._sweeps == 1
    >>> assert len(repeater.data((1, 10))) == 0
    >>> assert len(repeater.evaluations((1, 2))) == 0

    """
    def __init__(self, budget_multiplier, min_successes=11):
        """``min_successes=11`` for instances 1-5 provokes at least three

        sweeps, hence 15 trials, given the `budget_multiplier` is large
        enough and the algorithm terminates early enough before the budget
        is exhausted. 3 x 1-5 is the instance-setup from BBOB 2009, however
        the choice of the initial solution still slightly differs.
        """
        self.params = {k: v for k, v in locals().items() if k != 'self'}
        self.max_sweeps = 1e4  # should not be necessary
        self._dimension_offset = 0
        '''budget = (dimension + offset) * budget_multiplier'''
        self._data = _collections.OrderedDict()
        '''a ``dict[(fun, dim)]`` of ``dict[iinst]`` of `list` of
           ``(evaluations, success)`` `tuples`'''
        self._calls = 0  # calls of `track`
        self._sweeps = 0  # calls of ``done()`` without argument
    def all(self, method):
        """return `dict` with ``method(key)`` as values and

        with all ``(id_function, dimension)`` as keys.

        Works with the instance methods `evaluations`, `successes`,
        `budget_exhausted`, `trials`, or `instances` as argument or the
        respective strings.
        """
        if method == str(method):
            method = getattr(self, method)
        return {(fun, dim): method((fun, dim)) for (fun, dim) in self._data}
    def budget_from_dimension(self, problem_or_dimension):
        """return budget_multiplier x ((problem.dimension or dimension) + self.offset).

        By default, ``offset == 0``.
        """
        
        dimension = problem_or_dimension if isinstance(problem_or_dimension, int) else (
                    self.problem_to_ids(problem_or_dimension)[1])
        return (dimension + self._dimension_offset) * self.params['budget_multiplier']
    def _empty_data(self, problem=None):
        """should always return an empty list, called in the `data` method"""
        if problem is None:
            return {p: self._empty_data(p) for p in self._data if self._empty_data(p)}
        fun, dim = self.problem_to_ids(problem)[:2]
        if (fun, dim) in self._data:
            d = self._data[(fun, dim)]
            return [ii for ii in d if not d[ii]]  # list of instance numbers with no data
        return []
    @property
    def n_problems(self):
        """current number of tracked problems (different ``(id_function, dimension)``)"""
        return sum(len(self.trials(k)) > 0 for k in self._data)
    def _n_instance_data(self, problem):
        return len(self.data_of_instance(problem))
    def n_problem_instances(self, problem=None):
        """number of problem instances (of `problem`) with at least one trial
        """
        if problem is not None:
            return sum(n > 0 for n in self.trials(problem))
            return len(self.data(problem))
            return sum(len(res_list) > 0 for res_list in self.data(problem).values())
        return sum(self.n_problem_instances(p) for p in self._data)
        return sum(len(d) for d in self._data.values())
    def n_trials(self, problem=None):
        """current number of tracked trials from all instances (of `problem`)"""
        if problem is not None:
            return sum(self.trials(problem))
        return sum(self.all(self.n_trials).values())
    def track(self, problem):
        """record problem instance evaluations and success (target hit)"""
        fun, dim, iinst = self.problem_to_ids(problem)
        if (fun, dim) not in self._data:
            self._data[(fun, dim)] = _collections.defaultdict(list)
        self._data[(fun, dim)][iinst] += [(problem.evaluations, problem.final_target_hit)]
        self._calls += 1
    def problem_to_ids(self, problem):
        """return ``(id_function, dimension, id_instance)`` of `problem`"""
        self._check(problem)
        if (not callable(problem)
                and isinstance(problem, (tuple, list))
                and len(problem) in (2, 3)):
            return tuple(problem) if len(problem) == 3 else (problem[0], problem[1], None)
        return problem.id_function, problem.dimension, problem.id_instance
    def _to_key(self, problem):
        """return `data`-key of problem, namely ``(id_function, dimension)``"""
        return self.problem_to_ids(problem)[:2]
    def _check(self, problem):
        """check problem to avoid kernel crashes, raise ValueError"""
        if hasattr(problem, 'info') and 'invalid' in problem.info:
            raise ValueError(problem.info)
    def _assert_consistencies(self):
        if self.done():  # the below is necessary but not sufficient to be done
            assert all(sum(v) for v in zip(self.all('budget_exhausted').values(),
                                           self.all('succeeded').values()))
        for (problem, data) in self._data.items():
            assert self.trials(problem) == [len(data[ii]) for ii in sorted(data)]
            assert sum(n > 0 for n in self.trials(problem)) == len(self.data(problem)), (
                    "found unexpected empty data in problem {}:"
                    "\n  trials=={}\n  data=={}"
                    .format(problem, self.trials(problem), self.data(problem)))
        assert (sum(len(d) for d in self._data.values()) == 
                sum(self.n_problem_instances(p) for p in self._data))
    def data(self, problem):
        """return reference to `dict` with instance as keys,

        where nonexisting entries are returned as empty but not permanently
        created.
        """
        if self._empty_data():
            _warnings.warn("empty data found when calling `data` method"
                           " with problem {}:\n  {}"
                           .format(self.problem_to_ids(problem), self._empty_data()))
        return self._data.get(self._to_key(problem), _collections.defaultdict(tuple))
        # return self._data.[(fun, dim)] if (fun, dim) in self._data else _collections.defaultdict(tuple)  # is minimally quicker
    def data_of_instance(self, problem):
        """return `list` of ``(evaluations, success)`` of problem instance"""
        return self.data(problem).get(self.problem_to_ids(problem)[2], ())
    def instances(self, problem):
        """return `list` of instance IDs tracked for (id_function, dimension) of `problem`.

        Instance IDs are the keys for the ``self.data(problem)`` dictionary.
        """
        return sorted(self.data(problem))  # unsorted dict keys have arbitrary order
    def trials(self, problem, instance=None):
        """return `list` of number of recorded trials for each instance

        if ``instance is None``, otherwise for the given `instance` only.
        """
        dd = self.data(problem)  # dictionary of all data related to problem
        if instance is not None:
            return len(dd[instance])
        return [len(dd[iinst]) for iinst in sorted(dd)]
    def successes(self, problem):
        """return `list` of successes per instance"""
        dd = self.data(problem)  # dictionary of all data related to problem
        return [sum(i[1] for i in dd[key]) for key in sorted(dd)]
    def evaluations(self, problem):
        """return `list` of summed evaluations per problem instance"""
        dd = self.data(problem)  # dictionary of all data related to problem
        return [sum(i[0] for i in dd[key]) for key in sorted(dd)]
    def budget_exhausted(self, problem):
        """compare average evaluations per instance with `budget_from_dimension`"""
        return sum(self.evaluations(problem)) >= max((1, len(self.instances(problem)))
                    ) * self.budget_from_dimension(problem)
    def succeeded(self, problem):
        """return `True` iff problem had at least `min_success` successes"""
        return sum(self.successes(problem)) >= self.params['min_successes']
    def message_sweep(self):
        """return status message and '' when nothing was tracked yet"""
        if not len(self._data) and self._sweeps <= 1:
            return ''
        successes = [sum(v) for v in self.all(self.successes).values()]
        succeeded = self.all(self.succeeded).values()
        return ('\nSweep {} done ({} success{} on {} problem{} where {} had >= 1'
            ' success and {} had >= {} successes and {} exhausted the budget)\n'
            .format(self._sweeps,
                    sum(successes),
                    'es' if sum(successes) != 1 else '',
                    len(self._data),
                    's' if len(self._data) != 1 else '',
                    sum(s > 0 for s in successes),
                    sum(succeeded),
                    self.params['min_successes'],
                    sum(b and not s for (s, b) in
                        zip(succeeded, self.all(self.budget_exhausted).values()))))
    def done(self, problem=None, message=True):
        """return `False` iff this/any `problem` instance remains to be (re-)run.

        When ``problem is not None``, return `True` if this problem
        instance does not require another trial, however `False` as long as
        `done` has not been called twice without argument (thereby assuming
        that the loop starts with ``while repeater.done()``).
 
        When ``problem is None``, return `True` if no single recorded problem
        requires another trial.

        Return invariably `False` before the second call of ``done()``
        without argument, which is considered to happen right before the
        second sweep when the first sweep is finished.

        Details
        -------
        ``done()`` without argument gives only consistent results before or
        after a _full_ first sweep. In particular, _during_ the first sweep
        it cannot account for problems that have not yet been run once.

        Calling ``done()`` increments the sweep counter iff it returns
        `False`, the default for ``max_sweeps`` is ``1e4``. The attribute
        can be directly reassigned at any time.

        See also: `remaining_problems`
        """
        if problem is not None:
            if self._sweeps <= 1 or self.missing_trials(problem) or (
                    not self.succeeded(problem)
                    and not self.budget_exhausted(problem)):
                return False
            return True
        if message:  # prints nothing when no data are found
            print(self.message_sweep(), end='')  # indicates "sequence of problems" case
        self._sweeps += 1  # for the above sweeps number check
        done = self._sweeps >= self.max_sweeps or (
            self.n_problem_instances() > 0 and not self.remaining_problems())
        self._sweeps -= done  # don't increment when done
        return done
    def remaining_problems(self, suite=None):
        """return `list` of (probably) remaining problem instances to run.

        `suite`, when given, compares the recorded experiments with the
        problem instances in the suite to determine what remains. The
        `suite` argument is bound to give unexpected results when some of
        the problems from `suite` are skipped to be run with other batches,
        e.g., in parallel.

        See also: `done`
        """
        if suite is not None:
            res = [self.problem_to_ids(p) for p in suite if not self.done(p)]
        else:
            res = [(fun, dim, iinst)
                        for (fun, dim), instances in self._data.items()
                            for iinst in instances
                                if not self.done((fun, dim, iinst))]
        return res
    def missing_trials(self, problem):
        """return number of missing trials for this problem instance

        compared to the other instances of the same problem or 1 when no
        trials were found at all.
        """
        return max((1, max([0] + self.trials(problem)))) - len(self.data_of_instance(problem))
    def initial_solution_proposal(self, problem, nonzero_odds=14):
        """return allzeros in the first of any ``nonzero_odds + 1`` trials

        if ``nonzero_odds > 0``, and otherwise
        ``problem.initial_solution_proposal(#trials_done + 1)``.
        """
        if nonzero_odds != int(nonzero_odds):
            raise ValueError("nonzero_odds={} must be an int".format(nonzero_odds))
        trials = sum(self.trials(problem))
        if nonzero_odds == 0:
            return problem.initial_solution_proposal(trials + 1)
        return problem.initial_solution_proposal(trials if trials % (nonzero_odds + 1) else 0)

class BatchScheduler:
    """Facilitate running a benchmarking experiment on a `cocoex.Suite` in
       several independent batches.

    The batch scheduler crucially assumes that in each batch the same
    problems are given _in the same order_ when calling `is_in_batch`.

    Pseudo code example::

        batch_to_execute = 0  # set current batch to execute in [0, 3]

        suite = cocoex.Suite('bbob', '', '')
        batcher = cocoex.BatchScheduler(4, batch_to_execute)
        for problem in suite:
            if not batcher.is_in_batch(problem):
                continue
            # ... run optimizer on problem ...
    
    needs to be run four times overall (e.g., in parallel) with
    ``batch_to_execute`` = 0..3 to generate the full experimental data.

    Details: to get a more even time distribution over all batches, it
    seems advisable that the number of functions is not divisible by the
    number of batches. That is, 4 (or 6 or 8 or 12) batches is not likely
    to be ideal on the `'bbob'` testbed of 24 functions.
    """
    def __init__(self, number_of_batches, batch_to_execute):
        """distribute over `number_of_batches` batches and executed here

        the batch with number `batch_to_execute`.
        """
        self.params = {n: v for (n, v) in locals().items() if n != 'self'}
        self.first_problem = None
        self.current_problem = None
        self.current_batch = -1
        if self.params['number_of_batches'] == 1 and self.params['batch_to_execute'] in (0, 1, None):
            print("number_of_batches == 1, hence running the full suite")
            self.params['batch_to_execute'] = 0
        elif self.params['number_of_batches'] <= self.params['batch_to_execute']:
            raise ValueError("number of batches == {} <= {} == batch to execute,"
                            " however > is required."
                            "\n  The first argument is the number of batches (nob),"
                            "\n  the second argument needs to sweep from 0...nob-1."
                            .format(self.params['number_of_batches'],
                                    self.params['batch_to_execute']))
    def is_in_batch(self, problem):
        """return `True` iff the batch number for `problem` equals `batch_to_execute`

        which was given as a constructor argument. Assumes that
        ``id_function`` and ``dimension`` are attributes of `problem`.

        The batch number for `problem` is attributed using
        ``(problem.id_function, problem.dimension)`` by order of
        appearance, in that the number is incremented whenever either
        ``id_function`` or ``dimension`` changes. Instances that follow
        suit with the same function ID and dimension belong to the same
        batch.
        """
        if self.first_problem is None:
            self.first_problem = (problem.id_function, problem.dimension)
        if self.current_problem != (problem.id_function, problem.dimension):
            if (problem.id_function, problem.dimension) == self.first_problem:
                self.current_batch = -1
            self.current_problem = (problem.id_function, problem.dimension)
            self.current_batch += 1
        return self.current_batch % self.params['number_of_batches'] == self.params['batch_to_execute']
