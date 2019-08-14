#!/usr/bin/env python
"""Create logger output for the bbob and bbob-biobj loggers when random search is run on the suite
with the same name.
"""
from __future__ import division, print_function
import numpy as np
import cocoex


def run_experiment(suite_name, logger, folder, order='', observer_options=''):
    """Runs random search on a subset of problems from either the bbob or bbob-biobj suites.

    If order='', the problems are called in the usual order.
    If order='rand', the problems are called in a random order.
    If order='inst', the problems are split wrt their instances (first half of instances goes first,
    then the other half).
    Reuses much of the code from the example experiment for beginners
    Returns the name of the actual output folder
    """
    ### input
    budget_multiplier = 50

    ### prepare
    suite = cocoex.Suite(suite_name, 'instances: 1-6', 'dimensions: 2,3 function_indices: 9-20')
    observer = cocoex.Observer(logger, 'result_folder: {} {}'.format(folder, observer_options))
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
        problem.observe_with(observer)
        # # Check the domain middle
        # print(problem(np.zeros(problem.dimension)))
        # # Check also NAN and INFINITE values
        # print(problem(np.nan * np.ones(problem.dimension)))
        # print(problem(np.inf * np.ones(problem.dimension)))
        while (problem.evaluations < problem.dimension * budget_multiplier
               and not problem.final_target_hit):
            x = np.random.uniform(problem.lower_bounds, problem.upper_bounds, problem.dimension)
            problem(x)
    return observer.result_folder


def run_two_observers(suite_name, logger, folder):
    """Uses two observers to observe the first problem of the given suite. Performs only a few
    random evaluations.

    Returns the name of the actual output folders
    """
    suite = cocoex.Suite(suite_name, 'instances: 1', 'dimensions: 2 function_indices: 1')
    observer1 = cocoex.Observer(logger, 'result_folder: {}-1'.format(folder))
    observer2 = cocoex.Observer(logger, 'result_folder: {}-2'.format(folder))

    np.random.seed(12345)

    problem = suite[0]
    problem.observe_with(observer1)
    problem.observe_with(observer2)
    for _ in range(10):
        x = np.random.uniform(problem.lower_bounds, problem.upper_bounds, problem.dimension)
        problem(x)

    return observer1.result_folder, observer2.result_folder


def run_several_problems(suite_name, logger, folder):
    """Works with several problems at the same time. Performs only a few random evaluations of these
    problems.

    Returns the name of the actual output folder
    """
    suite = cocoex.Suite(suite_name, 'instances: 1-2', 'dimensions: 2,3 function_indices: 1-2')
    observer = cocoex.Observer(logger, 'result_folder: {}'.format(folder))

    np.random.seed(12345)

    problems = [suite[i] for i in range(len(suite))]
    for problem in problems:
        problem.observe_with(observer)
    for _ in range(10):
        for problem in problems:
            x = np.random.uniform(problem.lower_bounds, problem.upper_bounds, problem.dimension)
            problem(x)

    return observer.result_folder


if __name__ == "__main__":
    # for logger in ['bbob-biobj']:
    for logger in ['bbob', 'bbob-biobj']:
        for order in ['default', 'rand', 'inst']:
            data_folder = '{}_logger_data_{}'.format(logger, order)
            run_experiment(logger, logger, data_folder, order=order)
            if order == 'default':
                run_experiment(
                    logger, logger, data_folder + '_options', order=order,
                    observer_options='unif_target_trigger: 1 unif_target_precision: 1e6')
        if logger is not 'bbob':
            run_two_observers(logger, logger, '{}_logger_data_{}'.format(logger, '2_observers'))
        # run_several_problems(logger, 'bbob-new', '{}_logger_data_{}'.format(logger, '8_problems'))
