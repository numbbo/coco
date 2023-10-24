#!/usr/bin/env python
"""DEPRECATED: use rather `example_experiment2.py`

A Python script for the COCO experimentation module `cocoex`.

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
>>> ee.observer_options['algorithm_info'] = '"default of example_experiment.py"'
>>> ee.main(5, 1+9, 2, 300)  # doctest: +ELLIPSIS
Benchmarking solver...

runs the 2nd of 300 batches with budget 5 * dimension and at most 9 restarts.

Calling `example_experiment` without parameters prints this
help and the available suite names.

DEPRECATED: use rather `example_experiment2.py`
"""
from __future__ import absolute_import, division, print_function
try: range = xrange
except NameError: pass
import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
import cocoex
from scipy import optimize # for tests with fmin_cobyla
from cocoex import Suite, Observer, log_level
del absolute_import, division, print_function

verbose = 1

try: import cma  # cma.fmin is a solver option, "pip install cma" installs cma
except: pass
try: from scipy.optimize import fmin_slsqp  # "pip install scipy" installs scipy
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass
try: time.process_time = time.clock
except: pass

from cocoex import default_observers  # see cocoex.__init__.py
from cocoex.utilities import ObserverOptions, ShortInfo, ascetime, print_flush
from cocoex.solvers import random_search

def default_observer_options(budget_=None, suite_name_=None, current_batch_=None):
    """return defaults computed from input parameters or current global vars
    """
    global budget, suite_name, number_of_batches, current_batch
    if budget_ is None:
        budget_ = budget
    if suite_name_ is None:
        suite_name_ = suite_name
    if current_batch_ is None and number_of_batches > 1:
        current_batch_ = current_batch
    opts = {}
    try:
        opts.update({'result_folder': '"%s_on_%s%s_budget%04dxD"'
                    % (SOLVER.__name__,
                       suite_name_,
                       "" if current_batch_ is None
                          else "_batch%03dof%d" % (current_batch_, number_of_batches),
                       budget_)})
    except: pass
    try:
        solver_module = '(%s)' % SOLVER.__module__
    except:
        solver_module = ''
    try:
        opts.update({'algorithm_name': SOLVER.__name__ + solver_module})
    except: pass
    return opts

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
        runs = coco_optimize(solver, problem, budget * problem.dimension, observer,
                             max_runs)
        if verbose:
            print_flush("!" if runs > 2 else ":" if runs > 1 else ".")
        short_info.add_evals(problem.evaluations + problem.evaluations_constraints, runs)
        problem.free()  # not necessary as `enumerate` tears the problem down
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
def coco_optimize(solver, fun, max_evals, observer, max_runs=1e9):
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
        observer.signal_restart(fun)

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
                x0 = "%f + %f * (np.random.rand(%d) - 0.5)" % (
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
            solver(fun, x0, lambda x: -fun.constraint(x), maxfun=remaining_evals,
                   disp=0, rhoend=1e-9)
############################ ADD HERE ########################################
        # ### IMPLEMENT HERE THE CALL TO ANOTHER SOLVER/OPTIMIZER ###
        # elif solver.__name__ == ...:
        #     CALL MY SOLVER, interfaces vary
##############################################################################
        else:
            solver(fun, x0)

        if fun.evaluations + fun.evaluations_constraints >= max_evals or \
           fun.final_target_hit:
            break
        # quit if fun.evaluations did not increase
        still_remaining = max_evals - fun.evaluations - fun.evaluations_constraints
        if still_remaining >= remaining_evals:  # break loop if no evaluations were done
            if still_remaining > remaining_evals:
                raise RuntimeError("function evaluations decreased")
            if still_remaining >= fun.dimension + 2:
                print("WARNING: %d evaluations of budget %d remaining" %
                      (still_remaining, max_evals))
            break
    return 1 + restarts  # number of (almost) independent launches of `solver`

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
# By default we call SOLVER(fun, x0), but the INTERFACE CAN BE ADAPTED TO EACH SOLVER ABOVE
SOLVER = random_search
# SOLVER = optimize.fmin_cobyla
# SOLVER = my_solver # SOLVER = fmin_slsqp # SOLVER = cma.fmin
suite_instance = "" # "year:2023"
suite_options = ""  # "dimensions: 2,3,5,10,20 "  # if 40 is not desired
# for more suite options, see http://numbbo.github.io/coco-doc/C/#suite-parameters
observer_options = ObserverOptions({  # is (inherited from) a dictionary
                    'algorithm_info': '"A SIMPLE RANDOM SEARCH ALGORITHM"', # CHANGE/INCOMMENT THIS!
                    # 'algorithm_name': '',  # default already provided from SOLVER name
                    # 'result_folder': '',  # default already provided from several global vars
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
    suite = Suite(suite_name, suite_instance, suite_options)

    observer_name = default_observers()[suite_name]
    # observer_name = another observer if so desired
    observer_options.update_gracefully(default_observer_options())
    observer = Observer(observer_name, observer_options.as_string)

    print("Benchmarking solver '%s' with budget=%d*dimension on %s suite, %s"
          % (' '.join(str(SOLVER).split()[:2]), budget,
             suite.name, time.asctime()))
    if number_of_batches > 1:
        print('Batch usecase, make sure you run *all* %d batches.\n' %
              number_of_batches)
    t0 = time.process_time()
    batch_loop(SOLVER, suite, observer, budget, max_runs,
               current_batch, number_of_batches)
    print(", %s (%s total elapsed time)." %
            (time.asctime(), ascetime(time.process_time() - t0)))
    print('Data written to folder', observer.result_folder)
    print('To post-process the data call \n'
          '    python -m cocopp %s \n'
          'from a system shell or \n'
          '    cocopp.main("%s") \n'
          'from a python shell' % (2 * (observer.result_folder,)))

# ===============================================
if __name__ == '__main__':
    """read input parameters and call `main()`"""
    if len(sys.argv) < 2 or sys.argv[1] in ["--help", "-h"]:
            print(__doc__)
            print("Recognized suite names: " + str(cocoex.known_suite_names))
            sys.exit(0)
    suite_name = sys.argv[1]
    if suite_name not in cocoex.known_suite_names:
        print('WARNING: "%s" not in known names %s' %
                (suite_name, str(cocoex.known_suite_names)))
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
