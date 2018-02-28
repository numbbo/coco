from __future__ import absolute_import, division, print_function
import sys as _sys  # avoid name space polluting
import time as _time
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


class ProblemNonAnytime(object):
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
            print("\nd={0}: ".format(self.dimension))
        print("{0}{1} ".format(
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

class MiniPrint(object):
    """print dimension when changed and a single symbol for each call.

    Details: print '|' if ``problem.final_target_hit`` else '.'
    """
    def __init__(self):
        self.dimension = None
    def __call__(self, problem, final=False):
        if self.dimension != problem.dimension:
            if self.dimension is not None:
                print('')
            print("%dD: " % problem.dimension, end='')
            self.dimension = problem.dimension
        print('|' if problem.final_target_hit else '.', end='')
        if final:  # final print
            print('')
        _sys.stdout.flush()

class ShortInfo(object):
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
        s = '\n    done in %.1e seconds/evaluation' % (self.evals_by_dimension[self.d_current])
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
        l = _time.asctime().split()
        d = l[0]
        d = l[1] + l[2]
        h, m, s = l[3].split(':')
        return d + ' ' + h + 'h' + m + ':' + s

def ascetime(sec):
    """return elapsed time as str.

    Example: return `"0h33:21"` if `sec == 33*60 + 21`. 
    """
    h = sec / 60**2
    m = 60 * (h - h // 1)
    s = 60 * (m - m // 1)
    return "%dh%02d:%02d" % (h, m, s)


def print_flush(*args):
    """print without newline but with flush"""
    print(*args, end="")
    _sys.stdout.flush()

del absolute_import, division, print_function

