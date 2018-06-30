#!/usr/bin/env python
"""A short and simple example experiment with restarts.

"""
from __future__ import division, print_function
import cocoex
from numpy.random import rand  # for randomised restarts
import cma

# input
suite_name = "rw-top-trumps"
cocoex.known_suite_names.append(suite_name)
output_folder = "cma-top-trumps"
budget_multiplier = 2  # increase to 10, 100, ...

# prepare
suite = cocoex.Suite(suite_name, "", "function_indices: 2-")
observer = cocoex.Observer("rw", "log_variables: none result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()

# go
for problem in suite:  # this loop will take several minutes or longer
    problem.observe_with(observer)  # generates the data for cocopp post-processing
    x0 = problem.initial_solution
    # apply restarts while neither the problem is solved nor the budget is exhausted
    while problem.evaluations < problem.dimension * budget_multiplier:
        # CMA-ES
        cma.fmin(problem, x0, 2, options=dict(maxfevals=problem.dimension * budget_multiplier -
                                                        problem.evaluations, verbose=-9))
        x0 = problem.lower_bounds + ((rand(problem.dimension) + rand(problem.dimension)) *
                    (problem.upper_bounds - problem.lower_bounds) / 2)
    minimal_print(problem, final=problem.index == len(suite) - 1)
