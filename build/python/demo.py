#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import coco
import numpy as np
try: range = xrange
except NameError: pass

MAXEVALS = 1e2

def my_optimizer(f, lower_bounds, upper_bounds, budget):
    n = len(lower_bounds)
    delta = upper_bounds - lower_bounds
    x_min = y_min = None
    for i in range(int(budget)):
        x = lower_bounds + np.random.rand(n) * delta
        y = f(x)
        if y_min is None or y < y_min:
            y_min = y
            x_min = x
    return x_min

if 1 < 3:
    # simplest use case 
    for problem in coco.Benchmark("bbob2009", "bbob2009_observer", "random_search"):
        my_optimizer(problem,
                     problem.lower_bounds,
                     problem.upper_bounds,
                     MAXEVALS)
        print("done with '%s' ... " % str(problem))
        problem.free()  # not strictly necessary (depends on the observer) 
    print("done.")

elif 1 < 3:
    # use case using problem_index which allows to pick and choose (e.g. to parallelize experiments) 
    bm = coco.Benchmark("bbob2009", "bbob2009_observer", "random_search")
    problem_index = 0
    while True:
        if 11 < 3 or ((problem_index + 0) % 5):
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
    # use case with "random access" via dimension, function, instance, method coco.problem_index is not implemented yet
    raise NotImplementedError
    bm = coco.Benchmark("bbob2009", "bbob2009_observer", "random_search")
    for dim in dimensions:
        for fun in functions:
            for instance in instances:
                problem_index = coco.problem_index(dim, fun, instance)
                problem = bm.get_problem(problem_index)
                if not problem:
                    print("fun %d instance %d in dimension %d not found" %
                          (fun, instance, dim))
                    continue
                my_optimizer(problem)
                problem.free()

