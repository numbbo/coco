#!/usr/bin/env python
"""Python script for the COCO experimentation module `cocoex`.

Usage from a system shell::

    python example_experiment.py bbob

runs a full but short experiment on the bbob suite. The optimization
algorithm used is determined by the `SOLVER` attribute in this file::

    python example_experiment.py bbob 20

runs the same experiment but with a budget of 20 * dimension
f-evaluations::

    python example_experiment.py bbob-biobj 1e3 1 20

runs the first of 20 batches with maximal budget of
1000 * dimension f-evaluations on the bbob-biobj suite.
All batches must be run to generate a complete data set.

Usage from a python shell:

>>> import example_experiment as ee
>>> ee.suite_name = "bbob-biobj"
>>> ee.SOLVER = ee.random_search  # which is default anyway
>>> ee.observer_options['algorithm_info'] = "default of example_experiment.py"
>>> ee.main(5, 1+9, 2, 300)  # doctest: +ELLIPSIS
Benchmarking solver...

runs the 2nd of 300 batches with budget 5 * dimension and at most 9 restarts.

Calling `example_experiment` without parameters prints this
help and the available suite names.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
try: range = xrange
except NameError: pass
import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
import cocoex
from scipy import optimize # for tests with fmin_cobyla
from cocoex import Suite, Observer, log_level
del absolute_import, division, print_function, unicode_literals

verbose = 1

try: import cma  # cma.fmin is a solver option, "pip install cma" installs cma
except: pass
try: from scipy.optimize import fmin_slsqp  # "pip install scipy" installs scipy
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass

def default_observers(update=None):
    """return a map from suite names to default observer names.

    This function can also be used to update this map using
    a `dict` or a `list` of key-value pairs.
    """
    # this is a function only to make the doc available and
    # because @property doesn't work on module level
    _default_observers.update(update or {})
    return _default_observers
_default_observers = {
    'bbob': 'bbob',
    'bbob-biobj': 'bbob-biobj',
    'bbob-biobj-ext': 'bbob-biobj',
    'bbob-constrained': 'bbob',
    'bbob-largescale': 'bbob',  # todo: needs to be confirmed
    }
def default_observer_options(budget_=None, suite_name_=None):
    """return defaults computed from input parameters or current global vars
    """
    global budget, suite_name
    if budget_ is None:
        budget_ = budget
    if suite_name_ is None:
        suite_name_ = suite_name
    opts = {}
    try:
        opts.update({'result_folder': '%s_on_%s_budget%04dxD'
                    % (SOLVER.__name__, suite_name_, budget_)})
    except: pass
    try:
        solver_module = '(%s)' % SOLVER.__module__
    except:
        solver_module = ''
    try:
        opts.update({'algorithm_name': SOLVER.__name__ + solver_module})
    except: pass
    return opts
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
        for c in ["u'", 'u"', "'", '"', "{", "}"]:
            s = s.replace(c, '')
        return s

def print_flush(*args):
    """print without newline but with flush"""
    print(*args, end="")
    sys.stdout.flush()


def ascetime(sec):
    """return elapsed time as str.

    Example: return `"0h33:21"` if `sec == 33*60 + 21`. 
    """
    h = sec / 60**2
    m = 60 * (h - h // 1)
    s = 60 * (m - m // 1)
    return "%dh%02d:%02d" % (h, m, s)


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
        self.t0_dimension = time.time()
        self.evals_dimension = 0
        self.evals_by_dimension = {}
        self.runs_function = 0
    def print(self, problem, end="", **kwargs):
        print(self(problem), end=end, **kwargs)
        sys.stdout.flush()
    def add_evals(self, evals, runs):
        self.evals_dimension += evals
        self.runs_function += runs
    def dimension_done(self):
        self.evals_by_dimension[self.d_current] = (time.time() - self.t0_dimension) / self.evals_dimension
        s = '\n    done in %.1e seconds/evaluation' % (self.evals_by_dimension[self.d_current])
        # print(self.evals_dimension)
        self.evals_dimension = 0
        self.t0_dimension = time.time()
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
        if self.f_current and f != self.f_current:
            res += self.function_done() + ' '
        if problem.dimension != self.d_current:
            res += '%s%s, d=%d, running: ' % (self.dimension_done() + "\n\n" if self.d_current else '',
                        ShortInfo.short_time_stap(), problem.dimension)
            self.d_current = problem.dimension
        if f != self.f_current:
            res += '%s' % f
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
        l = time.asctime().split()
        d = l[0]
        d = l[1] + l[2]
        h, m, s = l[3].split(':')
        return d + ' ' + h + 'h' + m + ':' + s

# ===============================================
# prepare (the most basic example solver)
# ===============================================
def random_search(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between `lbounds` and `ubounds`."""
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
    max_chunk_size = 1 + 4e4 / dim
    while budget > 0:
        chunk = int(min([budget, max_chunk_size]))
        # about five times faster than "for k in range(budget):..."
        X = lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim)
        if fun.number_of_constraints > 0:
            C = [fun.constraint(x) for x in X]
        F = [fun(x) for x in X]
        if fun.number_of_objectives == 1:
            index = np.argmin(F)
            if f_min is None or F[index] < f_min:
                x_min, f_min = X[index], F[index]
        budget -= chunk
    return x_min

# ===============================================
# loops over a benchmark problem suite
# ===============================================
def batch_loop(solver, suite, observer, budget,
               max_runs, current_batch, number_of_batches):
    """loop over all problems in `suite` calling
    `coco_optimize(solver, problem, budget * problem.dimension, max_runs)`
    for each eligible problem.

    A problem is eligible if ``problem_index + current_batch - 1``
    modulo ``number_of_batches`` equals ``0``.

    This distribution into batches is likely to lead to similar
    runtimes for the batches, which is usually desirable.
    """
    addressed_problems = []
    short_info = ShortInfo()
    for problem_index, problem in enumerate(suite):
        if (problem_index + current_batch - 1) % number_of_batches:
            continue
        observer.observe(problem)
        short_info.print(problem) if verbose else None
        runs = coco_optimize(solver, problem, budget * problem.dimension,
                             max_runs)
        if verbose:
            print_flush("!" if runs > 2 else ":" if runs > 1 else ".")
        short_info.add_evals(problem.evaluations + problem.evaluations_constraints, runs)
        problem.free()
        addressed_problems += [problem.id]
    print(short_info.function_done() + short_info.dimension_done())
    short_info.print_timings()
    print("  %s done (%d of %d problems benchmarked%s)" %
           (suite_name, len(addressed_problems), len(suite),
             ((" in batch %d of %d" % (current_batch, number_of_batches))
               if number_of_batches > 1 else "")), end="")
    if number_of_batches > 1:
        print("\n    MAKE SURE TO RUN ALL BATCHES", end="")
    return addressed_problems

#===============================================
# interface: ADD AN OPTIMIZER BELOW
#===============================================
def coco_optimize(solver, fun, max_evals, max_runs=1e9):
    """`fun` is a callable, to be optimized by `solver`.

    The `solver` is called repeatedly with different initial solutions
    until either the `max_evals` are exhausted or `max_run` solver calls
    have been made or the `solver` has not called `fun` even once
    in the last run.

    Return number of (almost) independent runs.
    """
    range_ = fun.upper_bounds - fun.lower_bounds
    center = fun.lower_bounds + range_ / 2
    if fun.evaluations:
        print('WARNING: %d evaluations were done before the first solver call' %
              fun.evaluations)

    for restarts in range(int(max_runs)):
        remaining_evals = max_evals - fun.evaluations - fun.evaluations_constraints
        x0 = center + (restarts > 0) * 0.8 * range_ * (
                np.random.rand(fun.dimension) - 0.5)
        fun(x0)  # can be incommented, if this is done by the solver

        if solver.__name__ in ("random_search", ):
            solver(fun, fun.lower_bounds, fun.upper_bounds,
                   remaining_evals)
        elif solver.__name__ == 'fmin' and solver.__globals__['__name__'] in ['cma', 'cma.evolution_strategy', 'cma.es']:
            if x0[0] == center[0]:
                sigma0 = 0.02
                restarts_ = 0
            else:
                x0 = "%f + %f * np.random.rand(%d)" % (
                        center[0], 0.8 * range_[0], fun.dimension)
                sigma0 = 0.2
                restarts_ = 6 * (observer_options.as_string.find('IPOP') >= 0)

            solver(fun, x0, sigma0 * range_[0], restarts=restarts_,
                   options=dict(scaling=range_/range_[0], maxfevals=remaining_evals,
                                termination_callback=lambda es: fun.final_target_hit,
                                verb_log=0, verb_disp=0, verbose=-9))
        elif solver.__name__ == 'fmin_slsqp':
            solver(fun, x0, iter=1 + remaining_evals / fun.dimension,
                   iprint=-1)
        elif solver.__name__ in ("fmin_cobyla", ):
            x0 = fun.initial_solution
            solver(fun, x0, lambda x: -fun.constraint(x), maxfun = remaining_evals)
############################ ADD HERE ########################################
        # ### IMPLEMENT HERE THE CALL TO ANOTHER SOLVER/OPTIMIZER ###
        # elif True:
        #     CALL MY SOLVER, interfaces vary
##############################################################################
        else:
            raise ValueError("no entry for solver %s" % str(solver.__name__))

        if fun.evaluations + fun.evaluations_constraints >= max_evals or \
           fun.final_target_hit:
            break
        # quit if fun.evaluations did not increase
        if fun.evaluations + fun.evaluations_constraints <= max_evals - remaining_evals:
            if max_evals - fun.evaluations - fun.evaluations_constraints > fun.dimension + 1:
                print("WARNING: %d evaluations remaining" %
                      remaining_evals)
            if fun.evaluations + fun.evaluations_constraints < max_evals - remaining_evals:
                raise RuntimeError("function evaluations decreased")
            break
    return restarts + 1

# ===============================================
# set up: CHANGE HERE SOLVER AND FURTHER SETTINGS AS DESIRED
# ===============================================
######################### CHANGE HERE ########################################
# CAVEAT: this might be modified from input args
suite_name = "bbob"  # always overwritten when called from system shell
                     # see available choices via cocoex.known_suite_names
budget = 2  # maxfevals = budget x dimension ### INCREASE budget WHEN THE DATA CHAIN IS STABLE ###
max_runs = 1e9  # number of (almost) independent trials per problem instance
number_of_batches = 1  # allows to run everything in several batches
current_batch = 1      # 1..number_of_batches
##############################################################################
SOLVER = random_search
# SOLVER = my_solver # SOLVER = fmin_slsqp # SOLVER = cma.fmin
suite_instance = "" # "year:2016"
suite_options = ""  # "dimensions: 2,3,5,10,20 "  # if 40 is not desired
# for more suite options, see http://numbbo.github.io/coco-doc/C/#suite-parameters
observer_options = ObserverOptions({  # is (inherited from) a dictionary
                    'algorithm_info': "A SIMPLE RANDOM SEARCH ALGORITHM", # CHANGE/INCOMMENT THIS!
                    # 'algorithm_name': "",  # default already provided from SOLVER name
                    # 'result_folder': "",  # default already provided from several global vars
                   })
######################### END CHANGE HERE ####################################

# ===============================================
# run (main)
# ===============================================
def main(budget=budget,
         max_runs=max_runs,
         current_batch=current_batch,
         number_of_batches=number_of_batches):
    """Initialize suite and observer, then benchmark solver by calling
    ``batch_loop(SOLVER, suite, observer, budget,...``
    """
    observer_name = default_observers()[suite_name]
    observer_options.update_gracefully(default_observer_options())

    observer = Observer(observer_name, observer_options.as_string)
    suite = Suite(suite_name, suite_instance, suite_options)

    print("Benchmarking solver '%s' with budget=%d*dimension on %s suite, %s"
          % (' '.join(str(SOLVER).split()[:2]), budget,
             suite.name, time.asctime()))
    if number_of_batches > 1:
        print('Batch usecase, make sure you run *all* %d batches.\n' %
              number_of_batches)
    t0 = time.clock()
    batch_loop(SOLVER, suite, observer, budget, max_runs,
               current_batch, number_of_batches)
    print(", %s (%s total elapsed time)." % 
            (time.asctime(), ascetime(time.clock() - t0)))

# ===============================================
if __name__ == '__main__':
    """read input parameters and call `main()`"""
    if len(sys.argv) < 2 or sys.argv[1] in ["--help", "-h"]:
            print(__doc__)
            print("Recognized suite names: " + str(cocoex.known_suite_names))
            exit(0)
    suite_name = sys.argv[1]
    if len(sys.argv) > 2:
        budget = float(sys.argv[2])
    if len(sys.argv) > 3:
        current_batch = int(sys.argv[3])
    if len(sys.argv) > 4:
        number_of_batches = int(sys.argv[4])
    if len(sys.argv) > 5:
        messages = ['Argument "%s" disregarded (only 4 arguments are recognized).' % sys.argv[i]
            for i in range(5, len(sys.argv))]
        messages.append('See "python example_experiment.py -h" for help.')
        raise ValueError('\n'.join(messages))
    main(budget, max_runs, current_batch, number_of_batches)
