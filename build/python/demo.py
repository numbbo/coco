#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from cocoex import Benchmark

try: import cma
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass

MAXEVALS = 1e2  # always start with something small

# prepare
def random_search(fun, lower_bounds, upper_bounds, budget):
    x_min = f_min = None
    for i in range(int(budget)):
        x = lower_bounds + np.random.rand(len(lower_bounds)) * (upper_bounds - lower_bounds)
        f = fun(x)
        if f_min is None or f < f_min:
            x_min, f_min = x, f
    return x_min

# set up
solver = random_search  # cma.fmin #
suite_name = "bbob2009"
suite_options = ""  # options syntax could be: "instances:1-5; dimensions:2-20",
observer_name = "bbob2009_observer"
observer_options = "%s_on_%s" % (solver.__name__, suite_name)  # TODO: "folder:random_search; verbosity:1"

# interface
def coco_solve(problem):
    # implement here the interface between the coco problem and given solver
    global solver 
    if solver.__name__ in ("random_search",):
        solver(problem, problem.lower_bounds, problem.upper_bounds,
                MAXEVALS)
        return
    if solver.__name__ == 'fmin' and solver.func_globals['__name__'] == 'cma':
        center = (problem.lower_bounds + problem.upper_bounds) / 2
        range_ = problem.upper_bounds - problem.lower_bounds
        solver(problem, center, 0.2, dict(scaling=range_, maxfevals=MAXEVALS, verbose=-9))

# run    
if 1 < 3:
    # simple Pythonic use case, never leaves a problem unfree()ed
    print('Pythonic usecase')
    for problem in Benchmark(suite_name, suite_options, observer_name, observer_options):
        # use problem under some conditions
        if 0 or ('f11' in problem.id and 'i03' in problem.id):
            coco_solve(problem)
    print("%s done." % suite_name)

if 1 < 3:
    # generic usecase possible in all languages
    print('Generic usecase')
    bm = Benchmark(suite_name, suite_options, observer_name, observer_options)  
    problem_index = bm.next_problem_index(-1)  # get first index, is not necessarily 0!
    while problem_index >= 0:
        problem = bm.get_problem(problem_index)  # this should give a console message by the observer
        # use problem under some conditions
        if 0 or ('i02' in problem.id and problem_index < 30):
            # print("on '%s' ... " % problem.id, end='')
            coco_solve(problem)
        problem.free()  # preferably free would not be necessary, but how?
        problem_index = bm.next_problem_index(problem_index)
    print("%s done." % suite_name)
 
if 11 < 3:
    # generic usecase, possible if solver can be cast into a coco_optimizer_t *
    # which might often not be a straight forward type conversion, because (i) the
    # optimizer takes a function (pointer) as input and (ii) argument passing to
    # the function might be impossible to negotiate
    print("Minimal usecase, doesn't work though")
    Benchmark(coco_solve, suite_name, suite_options, observer_name, observer_options)

