#!/usr/bin/env python
"""Use case for the COCO experimentation module `cocoex` which can be used as
template.

Usage from a system shell::

    python example_experiment.py 3 1 20

runs the first of 20 batches with maximal budget
of 3 * dimension f-evaluations.

Usage from a python shell::

    >>> import example_experiment as ee
    >>> ee.main(3, 1, 1)  # doctest: +ELLIPSIS
    Benchmarking solver...

does the same but runs the "first" of one single batch.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
from cocoex import Suite, Observer

try: import cma  # cma.fmin is a solver option, "pip install cma" installs cma
except: pass
try: from scipy.optimize import fmin_slsqp  # "pip install scipy" installs scipy
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass


class PrintShortInfo(object):
    """print minimal info during benchmarking.
    
    After initialization, to be called right before the solver is called with
    the respective problem.
    
    Example output:
    
        Tue 22h39:02, d=2, running: f01 f02 f03 f04 f05 f06 f07 f08 f09 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26 f27 f28 f29 f30 f31 f32 f33 f34 f35 f36 f37 f38 f39 f40 f41 f42 f43 f44 f45 f46 f47 f48 f49 f50 f51 f52 f53 f54 f55 done
        Tue 22h39:03, d=3, running: f01 f02 f03 f04 f05 f06 f07 f08 f09 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26 f27 f28 f29 f30 f31 f32 f33 f34 f35 f36 f37 f38 f39 f40 f41 f42 f43 f44 f45 f46 f47 f48 f49 f50 f51 f52 f53 f54 f55 done
        Tue 22h39:03, d=5, running: f01 f02 f03 f04 f05 f06 f07 f08 f09 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26 f27 f28 f29 f30 f31 f32 f33 f34 f35 f36 f37 f38 f39 f40 f41 f42 f43 f44 f45 f46 f47 f48 f49 f50 f51 f52 f53 f54 f55 done
        Tue 22h39:04, d=10, running: f01 f02 f03 f04 f05 f06 f07 f08 f09 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26 f27 f28 f29 f30 f31 f32 f33 f34 f35 f36 f37 f38 f39 f40 f41 f42 f43 f44 f45 f46 f47 f48 f49 f50 f51 f52 f53 f54 f55 done
        Tue 22h39:04, d=20, running: f01 f02 f03 f04 f05 f06 f07 f08 f09 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26 f27 f28 f29 f30 f31 f32 f33 f34 f35 f36 f37 f38 f39 f40 f41 f42 f43 f44 f45 f46 f47 f48 f49 f50 f51 f52 f53 f54 f55 done
        Tue 22h39:05, d=40, running: f01 f02 f03 f04 f05 f06 f07 f08 f09 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26 f27 f28 f29 f30 f31 f32 f33 f34 f35 f36 f37 f38 f39 f40 f41 f42 f43 f44 f45 f46 f47 f48 f49 f50 f51 f52 f53 f54 f55 done

    """
    def __init__(self):
        self.f_current = 0  # function id (not problem id)
        self.d_current = 0  # dimension
    def __call__(self, problem):
        """uses `problem.id` and `problem.dimension` to decide what to print.
        """
        f = "f" + problem.id.lower().split('_f')[1].split('_')[0]
        if problem.dimension != self.d_current:
            print('%s%s, d=%d, running: ' % ('done\n\n' if self.d_current else '',
                        self.short_time_stap(), problem.dimension), end="")
            self.d_current = problem.dimension
        if f != self.f_current:
            print(' %s' % f, end='')
            self.f_current = f
        sys.stdout.flush()
    def short_time_stap(self):
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
def simple_loop(solver, suite, observer, budget_multiplier):
    """loop over all problems in `suite` calling `solver` with
    max budget `budge_multipier * dimension`.
    """
    found_problems, addressed_problems = 0, 0
    print_short_info = PrintShortInfo()
    for problem in suite:
        found_problems += 1
        # use problem only under some conditions, mainly for testing
        if 11 < 3 and not ('f11' in problem.id and 'i03' in problem.id):
            continue
        observer.observe(problem)
        print_short_info(problem)
        coco_optimize(solver, problem, budget_multiplier * problem.dimension)
        print(".", end="")
        addressed_problems += 1
    print("done\n%s done (%d of %d problems benchmarked)"
          % (suite_name, addressed_problems, found_problems), end="")


def batch_loop(solver, suite, observer, budget_multiplier,
               current_batch, number_of_batches):
    """loop over some problems in `suite` calling `solver` with
    max budget `budge_multipier * dimension`.

    A problem is executed if `number_of_batches` is one or if
    `problem_index + current_batch` modulo `number_of_batches` equals to one.
    """
    addressed_problems = []
    print_short_info = PrintShortInfo()
    for problem_index, problem_id in enumerate(suite.ids):
        if (problem_index + current_batch - 1) % number_of_batches:
            continue
        problem = suite.get_problem(problem_index, observer)
        print_short_info(problem)
        coco_optimize(solver, problem, budget_multiplier * problem.dimension)
        print(".", end="")
        problem.free()
        addressed_problems += [problem_id]
    print("%s done (%d of %d problems benchmarked%s)" %
           (suite_name, len(addressed_problems), len(suite),
             ((" in batch %d of %d" % (current_batch, number_of_batches))
               if number_of_batches > 1 else "")), end="")

#===============================================
# interface: ADD AN OPTIMIZER BELOW
#===============================================
def coco_optimize(solver, fun, budget):
    """`fun` is a callable, to be optimized by `solver`.

    The `solver` is called repeatedly with different initial solutions
    until the budget is exhausted.
    """
    range_ = fun.upper_bounds - fun.lower_bounds
    center = fun.lower_bounds + range_ / 2
    dim = len(fun.lower_bounds)

    runs = 0
    while budget > fun.evaluations:
        runs += 1
        remaining_budget = budget - fun.evaluations
        x0 = center if fun.evaluations == 0 else \
             center + 0.8 * range_ * (np.random.rand(dim) - 0.5)

        if solver.__name__ in ("random_search", ):
            solver(fun, fun.lower_bounds, fun.upper_bounds,
                    remaining_budget)
        elif solver.__name__ == 'fmin' and solver.func_globals['__name__'] == 'cma':
            # x0 = "%f + %f * np.rand(%d)" % (center[0], range_[0], dim)  # for bbob
            solver(fun, x0, 0.2 * range_, restarts=8,
                   options=dict(scaling=range_, maxfevals=remaining_budget,
                                verbose=-9))
        elif solver.__name__ == 'fmin_slsqp':
            solver(fun, x0, iter=1 + remaining_budget / dim, iprint=-1)
############################ ADD HERE ########################################
        # ### IMPLEMENT HERE THE CALL TO ANOTHER SOLVER/OPTIMIZER ###
        # elif True:
        #     CALL MY SOLVER, interfaces vary
##############################################################################
        else:
            print("no entry for solver %s" % str(solver.__name__))

        if fun.number_of_objectives == 1 and \
                fun.best_observed_fvalue1 < fun.final_target_fvalue1:
            break
    if runs > 1:
        print("%d runs, " % runs, end="")

# ===============================================
# set up: CHANGE HERE SOLVER AND FURTHER SETTINGS AS DESIRED
# ===============================================
######################### CHANGE HERE ########################################
SOLVER = random_search
#SOLVER = my_solver # fmin_slsqp # cma.fmin #
suite_name = "bbob-biobj"
# suite_name = "bbob"
suite_instance = ""  # 'dimensions: 2,3,5,10,20 instance_idx: 1-5'
suite_options = ""
observer_name = suite_name
observer_options = (
    ' result_folder: ' + os.path.join('exdata', '%s_on_%s ' % (SOLVER.__name__, suite_name)) +
    ' log_level: warning ' +
    ' algorithm_name: %s ' % SOLVER.__name__ +
    ' algorithm_info: "A SIMPLE RANDOM SEARCH ALGORITHM" ')  # CHANGE THIS

######################### CHANGE HERE ########################################
# CAVEAT: this might be modified from input args
budget_multiplier = 2  # times dimension ### INCREASE THE MULTIPLIER WHEN THE DATA CHAIN IS STABLE ###
number_of_batches = 1  # allows to run everything in several batches
current_batch = 1      # 1..number_of_batches
##############################################################################

# ===============================================
# run (main)
# ===============================================
def main(budget_multiplier=budget_multiplier,
         current_batch=current_batch,
         number_of_batches=number_of_batches):
    print("Benchmarking solver '%s' with budget=%d * dimension"
          % (' '.join(str(SOLVER).split()[:2]), budget_multiplier), end='')
    observer = Observer(observer_name, observer_options)
    suite = Suite(suite_name, suite_instance, suite_options)
    print(" on suite %s, %s" % (suite.name, time.asctime()))
    t0 = time.clock()
    if 1 < 3:
        print('Simple usecase ...'); sys.stdout.flush()
        simple_loop(SOLVER, suite, observer, budget_multiplier)
    elif 1 < 3:
        print('Batch usecase ...'); sys.stdout.flush()
        batch_loop(SOLVER, suite, observer, budget_multiplier,
                   current_batch, number_of_batches)
    print(", %s (%.2f min)." % (time.asctime(), (time.clock()-t0)/60**1))

# ===============================================
if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print(__doc__)
            exit(0)
        budget_multiplier = float(sys.argv[1])
    if len(sys.argv) > 2:
        current_batch = int(sys.argv[2])
    if len(sys.argv) > 3:
        number_of_batches = int(sys.argv[3])
    if len(sys.argv) > 4:
        messages = ['Argument "%s" disregarded (only 3 arguments are recognized).' % sys.argv[i]
            for i in range(4, len(sys.argv))]
        messages.append('See "python example_experiment.py -h" for help.')
        raise ValueError('\n'.join(messages))
    main(budget_multiplier, current_batch, number_of_batches)
