#!/usr/bin/env python
"""Create logger output for the bbob and bbob-biobj loggers when random search is run on the suite
with the same name.
"""
from __future__ import division, print_function
import os
import numpy as np
import cocoex


def run_experiment(logger, folder):
    """Runs random search on a subset of problems from either the bbob or bbob-biobj suites.

    Reuses much of the code from the example experiment for beginners
    """
    ### input
    suite_name = logger
    budget_multiplier = 5

    ### prepare
    suite = cocoex.Suite(suite_name, 'instances: 5-6,5-6', 'dimensions: 2,3 function_indices: 9-20')
    observer = cocoex.Observer(logger, 'result_folder: {}'.format(folder))
    minimal_print = cocoex.utilities.MiniPrint()
    np.random.seed(12345)

    ### go
    for problem in suite:  # this loop will take several minutes or longer
        problem.observe_with(observer)  # generates the data for cocopp post-processing
        # apply restarts while neither the problem is solved nor the budget is exhausted
        while (problem.evaluations < problem.dimension * budget_multiplier
               and not problem.final_target_hit):
            x = np.random.uniform(problem.lower_bounds, problem.upper_bounds, problem.dimension)
            problem(x)
        minimal_print(problem, final=problem.index == len(suite) - 1)


if __name__ == "__main__":
    for logger in ['bbob', 'bbob-biobj']:
        data_folder = os.path.join('..', '..', 'data', '{}_logger_data'.format(logger))
        run_experiment(logger, data_folder)

