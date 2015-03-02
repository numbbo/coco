#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import numpy as np  # "pip install numpy" installs numpy
from cocoex import Benchmark

try: import cma  # cma.fmin is a solver option, "pip install cma" installs cma
except: pass
try: from scipy.optimize import fmin_slsqp  # "pip install scipy" installs scipy
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass

"""Timing of random_search (cma.fmin is 100x slower) on a MacBook Pro 2014 [s]: 
    evals =       [ 1e2, 1e3, 1e4, 2e4, 4e4  ],  # on 2160 problems
    old_code =    [ -1,  -1,  172,           ],
    time_pre_opt =[  3,  20,  171, 336, 657  ],
    time =        [  2,  13,  101, 201,      ],
    time_in_C =   [  2,  10,   80, 152, 299, ],
    time_cma =    [168, 444, 1417,]  # 0.05 millisecond per feval
    time_slsqp =  [  -1, ]  # requires restart for a fair measurement
"""

# prepare (the most basic example solver)
def random_search(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between `lbounds` and `ubounds`. """
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
solver = random_search # cma.fmin # fmin_slsqp #   
suite_name = "bbob2009"
suite_options = ""  # options syntax could be: "instances:1-5; dimensions:2-20",
observer_name = "bbob2009_observer"
observer_options = "%s_on_%s" % (solver.__name__, suite_name)  # TODO: "folder:random_search; verbosity:1"
number_of_batches = 99  # CAVEAT: this might be modified below from input args
current_batch = 1

# interface
def coco_solve(fun):
    """fun is a callable, to be optimized by global variable `solver`"""
    range_ = fun.upper_bounds - fun.lower_bounds
    center = fun.lower_bounds + range_ / 2
    dim = len(fun.lower_bounds)

    # ENHANCE: restarts from randomized x0 points
    # NEEDED: know the remaining budget
    global solver 
    if solver.__name__ in ("random_search", ):
        solver(fun, fun.lower_bounds, fun.upper_bounds,
                MAXEVALS)
    elif solver.__name__ == 'fmin' and solver.func_globals['__name__'] == 'cma':
        solver(fun, center, 0.2, restarts=8, options=dict(scaling=range_, maxfevals=MAXEVALS, verbose=-9))
    elif solver.__name__ == 'fmin_slsqp':
        solver(fun, center, iter=MAXEVALS / dim + 1, iprint=-1)
    # IMPLEMENT HERE the call to given solver
    # elif ...:

# run 
if __name__ == '__main__':
    if len(sys.argv) > 1:
        MAXEVALS = float(sys.argv[1])
    if len(sys.argv) > 2:
        current_batch = int(sys.argv[2])
    if len(sys.argv) > 3:
        number_of_batches = int(sys.argv[3])
        
    if 11 < 3:
        # simple Pythonic use case, never leaves a problem unfree()ed, ctrl-C "safe"
        print('Pythonic usecase ...'); sys.stdout.flush()
        found_problems, addressed_problems = 0, 0
        for problem in Benchmark(suite_name, suite_options, 
                                 observer_name, observer_options):
            found_problems += 1
            # use problem only under some conditions, mainly for testing
            if 1 or ('f11' in problem.id and 'i03' in problem.id):
                coco_solve(problem)
                addressed_problems += 1
        print("%s done (%d of %d problems benchmarked)." 
                % (suite_name, addressed_problems, found_problems))
    
    elif 1 < 3:
        # usecase with batches
        print('Batch usecase ...'); sys.stdout.flush()
        bm = Benchmark(suite_name, suite_options, observer_name, observer_options)  
        addressed_problems = []
        for problem_index in bm.problem_indices: # bm.next_problem_index(problem_index) is also available
            if (problem_index + current_batch - 1) % number_of_batches:
                continue
            problem = bm.get_problem(problem_index) 
            coco_solve(problem)
            print("%d evaluations done (according to Python interface)" % problem.evaluations)
            problem.free()  # preferably free would not be necessary, but how?
            addressed_problems += [problem_index]
        print("%s done (%d of %d problems benchmarked%s)." % 
               (suite_name, len(addressed_problems), len(bm),
                 ((" in batch %d of %d" % (current_batch, number_of_batches))
                   if number_of_batches > 1 else "")))

    elif 1 < 3:
        # generic example, similarly possible in all languages, with batches
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
                coco_solve(problem)
                addressed_problems += 1
            problem.free()  # preferably free would not be necessary, but how?
        print("%s done (%d of %d problems benchmarked%s)." % 
               (suite_name, addressed_problems, found_problems,
                 ((" in batch %d of %d" % (current_batch, number_of_batches))
                   if number_of_batches > 1 else "")))
     
    if 11 < 3:
        # generic usecase, possible if solver can be cast into a coco_optimizer_t *
        # which might often not be a straight forward type conversion, because (i) the
        # optimizer takes a function (pointer) as input and (ii) argument passing to
        # the function might be impossible to negotiate
        print("Minimal usecase, doesn't work though")
        Benchmark(coco_solve, suite_name, suite_options, observer_name, observer_options)
        
