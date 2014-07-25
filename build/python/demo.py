#!/usr/bin/env python

import coco
import numpy as np

def my_optimizer(f, lower_bounds, upper_bounds, budget):
    n = len(lower_bounds)
    delta = upper_bounds - lower_bounds
    x = lower_bounds + np.random.rand(n) * delta
    for i in range(budget):
        y = f(x)
        
for problem in coco.Benchmark("toy_suit", "toy_observer", "random_search"):
    print "Optimizing '%s' ... " % str(problem)
    my_optimizer(problem,
                 problem.lower_bounds,
                 problem.upper_bounds,
                 100000)
    problem.free()
