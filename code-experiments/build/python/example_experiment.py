#!/usr/bin/env python
"""Use cases for `cocoex` which can be used as templates.

Usage from a system shell::

    [python] ./example_experiment.py 100 1 20

runs the first of 20 batches with maximal budget of 100 f-evaluations.

Usage from a python shell::

    >>> import example_experiment as example
    >>> example.main(100, 1, 1)  # doctest: +ELLIPSIS
    Benchmarking solver '<function random_search' with MAXEVALS=100, ...
    Batch usecase ...
    suite_biobj done (1650 of 1650 problems benchmarked), ...

does the same but runs the "first" of one single batch.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import time
import numpy as np  # "pip install numpy" installs numpy
from cocoex import Suite, Observer

try: import cma  # cma.fmin is a solver option, "pip install cma" installs cma
except: pass
try: from scipy.optimize import fmin_slsqp  # "pip install scipy" installs scipy
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass

#################################################
# prepare (the most basic example solver)
#################################################
def random_search(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between `lbounds` and `ubounds`."""
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
    max_chunk_size = 1 + 4e4 / dim
    while budget > 0:
        chunk = int(min([budget, max_chunk_size]))
        # about five times faster than "for k in range(budget):..."
        X = lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim)
        F = [fun(x) for x in X]
        index = np.argmin(F)
        if f_min is None or F[index] < f_min:
            x_min, f_min = X[index], F[index]
        budget -= chunk
    return x_min
    
#################################################
# set up
#################################################
MAXEVALS = 1e2  # always start with something small, CAVEAT: this might be modified from input args
solver = random_search # fmin_slsqp # cma.fmin #    
suite_name = "suite_bbob"
suite_name = "suite_biobj"
suite_instance = ""  # syntax see C code TODO
suite_options = ""
observer_name = "observer_bbob"
observer_name = "observer_biobj"
observer_options = "result_folder: %s_on_%s" % (solver.__name__, suite_name) 
number_of_batches = 1  # CAVEAT: this can be modified below from input args
current_batch = 1       # ditto

#################################################
# interface: add an optimizer here
#################################################
def coco_optimize(fun, budget=MAXEVALS):
    """fun is a callable, to be optimized by global variable `solver`"""
    range_ = fun.upper_bounds - fun.lower_bounds
    center = fun.lower_bounds + range_ / 2
    dim = len(fun.lower_bounds)

    runs = 0
    while budget > fun.evaluations:
        # print("%f %f" % (fun.best_observed_fvalue1, fun.final_target_fvalue1))
        if fun.best_observed_fvalue1 < fun.final_target_fvalue1:
            break
        remaining_budget = budget - fun.evaluations
        x0 = center + (fun.evaluations > 0) * 0.9 * range_ * (np.random.rand(dim) - 0.5)

        global solver 
        if solver.__name__ in ("random_search", ):
            solver(fun, fun.lower_bounds, fun.upper_bounds,
                    remaining_budget)
        elif solver.__name__ == 'fmin' and solver.func_globals['__name__'] == 'cma':
            # x0 = "%f + %f * np.rand(%d)" % (center[0], range_[0], dim)  # for bbob
            solver(fun, x0, 0.2, restarts=8,
                   options=dict(scaling=range_, maxfevals=remaining_budget, verbose=-9))
        elif solver.__name__ == 'fmin_slsqp':
            solver(fun, x0, iter=1 + remaining_budget / dim, iprint=-1)
        # IMPLEMENT HERE the call to given solver
        # elif ...:
        else:
            print("no entry for solver %s" % str(solver.__name__))
        runs += 1
    if runs > 1:
        print("%d runs, " % runs, end="")

#################################################
# run 
#################################################
def main(MAXEVALS=MAXEVALS, current_batch=current_batch, number_of_batches=number_of_batches):
    print("Benchmarking solver '%s' with MAXEVALS=%d, %s" % (' '.join(str(solver).split()[:2]), MAXEVALS, time.asctime(), ))
    suite = Suite(suite_name, suite_instance, suite_options)
    observer = Observer(observer_name, observer_options)
    t0 = time.clock()
    if 11 < 3:  # crashes due to memory allocation/free in next_problem() of coco_suite.c
        # simple Pythonic use case
        print('Pythonic usecase ...'); sys.stdout.flush()
        found_problems, addressed_problems = 0, 0
        for problem in suite:
            found_problems += 1
            # use problem only under some conditions, mainly for testing
            if 11 < 3 and not ('f11' in problem.id and 'i03' in problem.id):
                continue
            observer.observe(problem)
            coco_optimize(problem, MAXEVALS)
            # problem.free()
            addressed_problems += 1
            # print(found_problems, addressed_problems); sys.stdout.flush()
        print("%s done (%d of %d problems benchmarked), %s (%f s)." 
                % (suite_name, addressed_problems, found_problems,
                   time.asctime(), (time.clock()-t0)/60**0))
    
    elif 1 < 3:
        # usecase with batches and observer
        print('Batch usecase ...'); sys.stdout.flush()
        addressed_problems = []
        for problem_index, id in enumerate(suite.ids):
            if (problem_index + current_batch - 1) % number_of_batches:
                continue
            problem = suite.next_problem(observer)
            # print("%4d: " % problem_index, end="")
            coco_optimize(problem, MAXEVALS)
            addressed_problems += [problem_index]
        print("%s done (%d of %d problems benchmarked%s), %s (%f min)." % 
               (suite_name, len(addressed_problems), len(suite),
                 ((" in batch %d of %d" % (current_batch, number_of_batches))
                   if number_of_batches > 1 else ""), time.asctime(), (time.clock()-t0)/60))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print(__doc__)
            exit(0)
        MAXEVALS = float(sys.argv[1])
    if len(sys.argv) > 2:
        current_batch = int(sys.argv[2])
    if len(sys.argv) > 3:
        number_of_batches = int(sys.argv[3])
    main(MAXEVALS, current_batch, number_of_batches)
