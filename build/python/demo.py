#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import numpy as np
from cocoex import Benchmark

try: import cma  # cma.fmin is a solver option
except: pass
try: from scipy.optimize import fmin_slsqp
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass

"""Timing of random_search on a MacBook Pro 2014 [s]: 
    evals=       [ 1e2, 1e3, 1e4, 2e4, 4e4  ],  # on 2160 problems
    old_code=    [ -1,  -1,  172,           ],
    time_pre_opt=[  3,  20,  171, 336, 657  ],
    time=        [  2,  13,  101, 201,      ],
    time_in_C=   [  2,  10,   80, 152, 299, ], 
"""

# prepare (the most basic example solver)
def random_search(fun, lbound, ubound, budget):
    # about five times faster than a "for k in range(budget)" implementation
    max_chunk_size = 4e4 / len(lbound)
    x_min = (lbound + ubound) / 2
    f_min = None
    while budget > 0:
        chunk = min([budget, max_chunk_size])
        X = lbound + (ubound - lbound) * np.random.rand(chunk, len(lbound))
        F = [fun(x) for x in X]
        index = np.argmin(F)
        if f_min is None or F[index] < f_min:
            x_min, f_min = X[index], F[index]
        budget -= chunk
    return x_min
    
# set up
MAXEVALS = 1e2  # always start with something small, CAVEAT: this might be modified below
solver = random_search # cma.fmin # fmin_slsqp # 
suite_name = "bbob2009"
suite_options = ""  # options syntax could be: "instances:1-5; dimensions:2-20",
observer_name = "bbob2009_observer"
observer_options = "%s_on_%s" % (solver.__name__, suite_name)  # TODO: "folder:random_search; verbosity:1"
number_of_batches = 1  # CAVEAT: this might be modified below
current_batch = 1

# interface
def coco_solve(problem):
    range_ = problem.upper_bounds - problem.lower_bounds
    center = problem.lower_bounds + range_ / 2
    dim = len(problem.lower_bounds)

    # ENHANCE: restarts from randomized x0 points
    global solver 
    if solver.__name__ in ("random_search", ):
        solver(problem, problem.lower_bounds, problem.upper_bounds,
                MAXEVALS)
    elif solver.__name__ == 'fmin' and solver.func_globals['__name__'] == 'cma':
        solver(problem, center, 0.2, dict(scaling=range_, maxfevals=MAXEVALS, verbose=-9))
    elif solver.__name__ == 'fmin_slsqp':
        solver(problem, center, iter=MAXEVALS / dim + 1, iprint=-1)
    # IMPLEMENT HERE the call of the coco problem by the given solver
    # elif ...:

# run 
if __name__ == '__main__':
    if len(sys.argv) > 1:
        MAXEVALS = float(sys.argv[1])
    if len(sys.argv) > 3:
        current_batch, number_of_batches = int(sys.argv[2]), int(sys.argv[3])
        
    if number_of_batches == 1:
        # simple Pythonic use case, never leaves a problem unfree()ed
        print('Pythonic usecase ...'); sys.stdout.flush()
        addressed_problems = 0
        for problem in Benchmark(suite_name, suite_options, observer_name, observer_options):
            # use problem only under some conditions, mainly for testing
            if 0 or ('f11' in problem.id and 'i03' in problem.id):
                coco_solve(problem)
                addressed_problems += 1
        print("%s done (%d problems benchmarked)." % (suite_name, addressed_problems))
    
    else:
        # usecase with batches
        print('Batch usecase ...'); sys.stdout.flush()
        bm = Benchmark(suite_name, suite_options, observer_name, observer_options)  
        addressed_problems = 0
        for problem_index in bm.problem_indices:  # bm.next_problem_index(problem_index) is also available
            if (problem_index + current_batch - 1) % number_of_batches:
                continue
            problem = bm.get_problem(problem_index) 
            coco_solve(problem)
            problem.free()  # preferably free would not be necessary, but how?
            addressed_problems += 1
        print("%s done (%d problems benchmarked in batch %d/%d)."
              % (suite_name, addressed_problems, current_batch, number_of_batches))

    if 11 < 3:
        # generic usecase possible in all languages, not ctrl-C "safe"
        print('Generic usecase ...'); sys.stdout.flush()
        bm = Benchmark(suite_name, suite_options, observer_name, observer_options)  
        addressed_problems = 0
        problem_index = bm.next_problem_index(-1)  # get first index, is not necessarily 0!
        while problem_index >= 0:
            if (problem_index + current_batch - 1) % number_of_batches:
                problem_index = bm.next_problem_index(problem_index)
                continue
            problem = bm.get_problem(problem_index) 
            # use problem only under some conditions, mainly for testing
            if 0 or ('d20' in problem.id and 'i01' in problem.id):
                coco_solve(problem)
                addressed_problems += 1
            problem.free()  # preferably free would not be necessary, but how?
            problem_index = bm.next_problem_index(problem_index)
        print("%s done (%d problems benchmarked)." % (suite_name, addressed_problems))
     
    if 11 < 3:
        # generic usecase, possible if solver can be cast into a coco_optimizer_t *
        # which might often not be a straight forward type conversion, because (i) the
        # optimizer takes a function (pointer) as input and (ii) argument passing to
        # the function might be impossible to negotiate
        print("Minimal usecase, doesn't work though")
        Benchmark(coco_solve, suite_name, suite_options, observer_name, observer_options)
        
        