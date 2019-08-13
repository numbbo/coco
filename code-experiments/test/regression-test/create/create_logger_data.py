#!/usr/bin/env python
"""Create logger output for the bbob and bbob-biobj loggers when random search is run on the suite
with the same name.
"""
from __future__ import division, print_function
import numpy as np
import cocoex


def run_experiment(suite_name, logger, folder, order=''):
    """Runs random search on a subset of problems from either the bbob or bbob-biobj suites.

    If order='', the problems are called in the usual order.
    If order='rand', the problems are called in a random order.
    If order='inst', the problems are split wrt their instances (first half of instances goes first,
    then the other half).
    Reuses much of the code from the example experiment for beginners
    Returns the name of the actual output folder
    """
    ### input
    budget_multiplier = 5

    ### prepare
    suite = cocoex.Suite(suite_name, 'instances: 1-6', 'dimensions: 2,3 function_indices: 9-20')
    observer = cocoex.Observer(logger, 'result_folder: {}'.format(folder))
    minimal_print = cocoex.utilities.MiniPrint()
    np.random.seed(12345)

    ### go
    problem_indices = np.arange(len(suite))
    if order == 'rand':
        np.random.shuffle(problem_indices)
    elif order == 'inst':
        num_dims = len(suite.dimensions)
        funcs = [x[x.find('_f')+1:x.find('_i')] for x in suite.ids()[0:int(len(suite)/num_dims)]]
        num_func = len(set(funcs))
        half_num_inst = int(len(suite) / num_func / num_dims / 2)
        other_half_num_inst = int((len(suite) - half_num_inst * num_func * num_dims) / num_func / num_dims)
        mask = np.tile(np.append(np.ones(half_num_inst, dtype=bool),
                                 np.zeros(other_half_num_inst, dtype=bool)),
                       num_dims * num_func)
        problem_indices = np.append(problem_indices[mask], problem_indices[~mask])
    for problem_index in problem_indices:
        problem = suite[problem_index]
        problem.observe_with(observer)  #
        while (problem.evaluations < problem.dimension * budget_multiplier
               and not problem.final_target_hit):
            x = np.random.uniform(problem.lower_bounds, problem.upper_bounds, problem.dimension)
            problem(x)
        minimal_print(problem, final=problem.index == len(suite) - 1)
    return observer.result_folder


if __name__ == "__main__":
    for logger in ['bbob', 'bbob-biobj']:
        for order in ['default', 'rand', 'inst']:
            data_folder = '{}_logger_data_{}'.format(logger, order)
            run_experiment(logger, logger, data_folder, order)

