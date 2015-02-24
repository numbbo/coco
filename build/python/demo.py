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
    # optimizer takes a function (pointer) as input and argument passing might be
    # impossible to negotiate
    print("Minimal usecase, doesn't work though")
    Benchmark(my_optimizer,  # see above
              "bbob2009", "instances:1-5",  # of 15 instances (not instance nb)
              "bbob20009_observer", "folder:random_search, verbosity:1")

if 1 < 3:
    # generic usecase possible in all languages
    print('Generic usecase')
    bm = Benchmark("bbob2009", "", # "instances:1-5, dimensions:2-20", 
                   "bbob2009_observer", "random_search") #"folder:random_search, verbosity:1")
    problem_index = bm.next_problem_index(-1)  # get first index, is not necessarily 0!
    while problem_index >= 0:
        problem = bm.get_problem(problem_index)  # this should give a console message by the observer
        if 'i02' in problem.id and problem_index < 30:
            print("on '%s' ... " % problem.id, end='')
            my_optimizer(problem, problem.lower_bounds, problem.upper_bounds,
                         MAXEVALS)
            print("done")  # to be removed when the observer is more verbose
        problem.free()  # this should give a console message by the observer, preferably free would not be necessary, but how?
        problem_index = bm.next_problem_index(problem_index)
 
if 1 < 3:
    # simple Pythonic use case, doesn't add much to the above but is safer
    print('Pythonic usecase')
    for problem in Benchmark("bbob2009", "", # TODO: here go the suit options
                             "bbob2009_observer", "random_search"):
        if 'f11' in problem.id and 'i03' in problem.id:
            print("on '%s' ... " % problem.id, end='')
            my_optimizer(problem, problem.lower_bounds, problem.upper_bounds,
                         MAXEVALS)
            print("done")  # to be removed when the observer is more verbose
        # problem.free()  # done in finalize of the generator Benchmark.__iter__ 
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

