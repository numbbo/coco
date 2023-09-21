#!/usr/bin/env python
"""Usage from a system shell::

    [python] ./demo.py 100 1 20
    
runs the first of 20 batches with maximal budget of 100 f-evaluations.

Usage from a python shell:: 

    >>> import demo
    >>> demo.main(100, 1, 99)
    
does the same from 99 batches. 

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys, time
import numpy as np  # "pip install numpy" installs numpy
from cocoex import Benchmark

try: import cma  # cma.fmin is a solver option, "pip install cma" installs cma
except: pass
try: from scipy.optimize import fmin_slsqp  # "pip install scipy" installs scipy
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass
try: time.process_time = time.clock
except: pass

"""Timing of random_search (cma.fmin is 100x slower) on a MacBook Pro 2014 [s]: 
    evals =       [ 1e2, 1e3, 1e4, 2e4, 4e4  ],  # on 2160 problems
    old_code =    [ -1,  -1,  172,           ],
    time_pre_opt =[  3,  20,  171, 336, 657  ],
    time =        [  2,  13,  101, 201,      ],
    time_in_C =   [  2,  10,   80, 152, 299, ],
    time_cma =    [179, 444, 1417, ]  # until budget (0.05 millisecond per feval)
    time_slsqp =  [  8,  48,  412, ]  # until final target
    time_slsqp =  [  8,  53,  491, ]  # until budget
"""

# prepare (the most basic example solver)
def random_search(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between `lbounds` and `ubounds`. """
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
    max_chunk_size = 1 + 4e4 / dim
    while budget > 0:
        chunk = min([budget, max_chunk_size])
        # about five times faster than "for k in range(budget):..."
        X = lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim)
        F = [fun(x) for x in X]
        index = np.argmin(F)
        if f_min is None or F[index] < f_min:
            x_min, f_min = X[index], F[index]
        budget -= chunk
    return x_min
    
# set up
MAXEVALS = 1e2  # always start with something small, CAVEAT: this might be modified from input args
solver = random_search # fmin_slsqp # cma.fmin #    
suite_name = "bbob"         # available suite names   matching observer names
observer_name = "bbob"      # -----------------------------------------------
                            # bbob                    bbob
                            # bbob-biobj              bbob-biobj
                            # bbob-biobj-ext          bbob-biobj
                            # bbob-largescale         bbob
                            # bbob-mixint             bbob
                            # bbob-biobj-mixint       bbob-biobj
suite_options = ""  # options syntax could be: "instances:1-5; dimensions:2-20",
observer_options = "%s_on_%s" % (solver.__name__, suite_name)  # TODO: "folder:random_search; verbosity:1"
# for more details on suite and observer options, see 
# http://numbbo.github.io/coco-doc/C/#suite-parameters and
# http://numbbo.github.io/coco-doc/C/#observer-parameters.
number_of_batches = 99  # CAVEAT: this might be modified below from input args
current_batch = 1       # ditto

# interface
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

# run 
def main(MAXEVALS=MAXEVALS, current_batch=current_batch, number_of_batches=number_of_batches):
    print("Benchmarking solver %s, %s" % (str(solver), time.asctime(), ))
    t0 = time.process_time()
    if 11 < 3:
        # simple Pythonic use case, never leaves a problem unfree()ed, ctrl-C "safe"
        print('Pythonic usecase ...'); sys.stdout.flush()
        found_problems, addressed_problems = 0, 0
        for problem in Benchmark(suite_name, suite_options, 
                                 observer_name, observer_options):
            found_problems += 1
            # use problem only under some conditions, mainly for testing
            if 11 < 3 and not ('f11' in problem.id and 'i03' in problem.id):
                continue
            coco_optimize(problem, MAXEVALS)
            addressed_problems += 1
        print("%s done (%d of %d problems benchmarked), %s (%f s)." 
                % (suite_name, addressed_problems, found_problems,
                   time.asctime(), (time.clock()-t0)/60**0))
    
    elif 1 < 3:
        # usecase with batches
        print('Batch usecase ...'); sys.stdout.flush()
        bm = Benchmark(suite_name, suite_options, observer_name, observer_options)  
        addressed_problems = []
        for problem_index in bm.problem_indices: # bm.next_problem_index(problem_index) is also available
            if (problem_index + current_batch - 1) % number_of_batches:
                continue
            problem = bm.get_problem(problem_index)
            # print("%4d: " % problem_index, end="")
            coco_optimize(problem, MAXEVALS)
            problem.free()  # preferably free would not be necessary, but how?
            addressed_problems += [problem_index]
        print("%s done (%d of %d problems benchmarked%s), %s (%f min)." % 
               (suite_name, len(addressed_problems), len(bm),
                 ((" in batch %d of %d" % (current_batch, number_of_batches))
                   if number_of_batches > 1 else ""), time.asctime(), (time.clock()-t0)/60))

    elif 1 < 3:
        # generic example with batches, similarly possible in all languages
        print('Generic usecase with batches...'); sys.stdout.flush()
        bm = Benchmark(suite_name, suite_options, observer_name, observer_options)  
        found_problems, addressed_problems = 0, 0
        problem_index = -1  # first index is not necessarily 0!
        while True:
            problem_index = bm.next_problem_index(problem_index)
            if problem_index < 0: 
                break 
            found_problems += 1
            if (problem_index + current_batch - 1) % number_of_batches:
                continue
            problem = bm.get_problem(problem_index) 
            # use problem only under some conditions, mainly for testing
            if 1 or ('d20' in problem.id and 'i01' in problem.id):
                print("%4d: " % problem_index, end="")
                coco_optimize(problem, MAXEVALS)
                addressed_problems += 1
            problem.free()  # preferably free would not be necessary
        print("%s done (%d of %d problems benchmarked%s), %s (%f min)." % 
               (suite_name, addressed_problems, found_problems,
                 ((" in batch %d of %d" % (current_batch, number_of_batches))
                   if number_of_batches > 1 else ""), time.asctime(), (time.clock()-t0)/60, ))
        
if __name__ == '__main__':
    if len(sys.argv) > 1:
        MAXEVALS = float(sys.argv[1])
    if len(sys.argv) > 2:
        current_batch = int(sys.argv[2])
    if len(sys.argv) > 3:
        number_of_batches = int(sys.argv[3])
    main(MAXEVALS, current_batch, number_of_batches)
