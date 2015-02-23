#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from cocoex import Benchmark
import numpy as np
try: range = xrange  # let range always be an iterator
except NameError: pass

MAXEVALS = 1e2

def my_optimizer(fun, lower_bounds, upper_bounds, budget):
    n = len(lower_bounds)
    delta = upper_bounds - lower_bounds
    x_min = f_min = None
    for i in range(int(budget)):
        x = lower_bounds + np.random.rand(n) * delta
        f = fun(x)
        if f_min is None or f < f_min:
            x_min, f_min = x, f
    return x_min

if 11 < 3:
    # generic usecase, possible if my_optimizer can be cast into a coco_optimizer_t *
    # which might often not be a straight forward type conversion, because the
    # optimizer takes a function (pointer) as input
    print('Smallest usecase')
    Benchmark(my_optimizer,
              "bbob2009", "instances:1-5",  # of 15 instances (not instance nb)
              "bbob20009_observer", "folder:random_search, verbosity:1")

if 1 < 3:
    # generic usecase possible in all languages
    # TODO: implement method next_problem_index
    print('Generic usecase')
    bm = Benchmark("bbob2009", "", # "instances:1-5, dimensions:2-20", 
                   "bbob2009_observer", "random_search") #"folder:random_search, verbosity:1")
    problem_index = bm.next_problem_index(-1)  # is not necessarily 0
    while problem_index >= 0:
        problem = bm.get_problem(problem_index)  # this should give a console message by the observer
        print("on '%s' ... " % (str(problem)), end='')
        my_optimizer(problem,
                     problem.lower_bounds,
                     problem.upper_bounds,
                     MAXEVALS)
        print("done")
        problem.free()  # this should give a console message by the observer
        # preferably free would not be necessary, but this might not be possible
        problem_index = bm.next_problem_index(problem_index)
 
if 11 < 3:
    # simple use case, somewhat Python specific, doesn't add much to the above
    # TODO: add suit options to this interface
    print('Pythonic usecase')
    for problem in Benchmark("bbob2009", "", # here go the suit options
                             "bbob2009_observer", "random_search"):
        print("on '%s' ... " % (str(problem)), end='')
        my_optimizer(problem,
                     problem.lower_bounds,
                     problem.upper_bounds,
                     MAXEVALS)
        print("done")
        problem.free()  # not strictly necessary (depends on the observer) 
    print("done.")

if 11 < 3:
    # depreciated
    # use case using problem_index which allows to pick and choose (e.g. to parallelize experiments) 
    print('Depreciated usecase')
    bm = Benchmark("bbob2009", "bbob2009_observer", "", "random_search")
    problem_index = 0
    while True:
        if 11 < 3 and ((problem_index + 0) % 5):
            problem_index += 1
            continue
            # problem.free()  # in case we need to move the test after get_problem
        problem = bm.get_problem(problem_index)
        if not problem:
            break
        my_optimizer(problem, problem.lower_bounds, problem.upper_bounds,
                     MAXEVALS)
        print("done with '%s' ... " % str(problem))
        problem.free()
        problem_index += 1
    print("done.")

if 11 < 3:
    # depreciated, the selection process should be implemented on the benchmark side
    # use case with "random access" via dimension, function, instance, method coco.problem_index is not implemented yet
    raise NotImplementedError
    bm = Benchmark("bbob2009", "bbob2009_observer", "random_search")
    dimensions = [2, 3, 5, 10, 20, 40] 
    functions = range(1, 25) 
    instances = np.r_[1:6, 31:41] 
    for dim in dimensions:
        for fun in functions:
            for instance in instances:
                problem_index = bm.problem_index(dim, fun, instance)
                problem = bm.get_problem(problem_index)
                if not problem:
                    print("fun %d instance %d in dimension %d not found" %
                          (fun, instance, dim))
                    continue
                my_optimizer(problem)
                problem.free()

