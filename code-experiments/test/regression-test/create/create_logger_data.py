#!/usr/bin/env python
"""Create logger data for random search run on a few evaluations on the suite with the given
options.
"""
from __future__ import division, print_function
import numpy as np
import cocoex
import os


def run_experiment(suite_name, suite_options, observer_name, observer_options, folder,
                   instance_order=''):
    """Runs random search on a subset of problems from the given suite.

    If order='', the problems are called in the usual order.
    If order='rand', the problems are called in a random order.
    If order='inst', the problems are split wrt their instances (first half of instances goes first,
    then the other half).

    Returns the name of the actual output folder
    """
    budget_multiplier = 5
    suite = cocoex.Suite(suite_name, '', suite_options)
    observer = cocoex.Observer(observer_name,
                               '{} result_folder: {}'.format(observer_options, folder))
    np.random.seed(12345)

    # Set the order of problem instances
    problem_indices = np.arange(len(suite))
    if instance_order == 'rand':
        np.random.shuffle(problem_indices)
    elif instance_order == 'inst':
        num_dims = len(suite.dimensions)
        funcs = [x[x.find('_f')+1:x.find('_i')] for x in suite.ids()[0:int(len(suite)/num_dims)]]
        num_func = len(set(funcs))
        half_num_inst = int(len(suite) / num_func / num_dims / 2)
        other_half_num_inst = \
            int((len(suite) - half_num_inst * num_func * num_dims) / num_func / num_dims)
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


def run_two_observers(suite_name, observer_name):
    """Uses two observers of the same kind to observe the first problem of the given suite.
    Performs only a few random evaluations.

    Returns the name of the actual output folders
    """
    folder = '2_observers_{}_{}'.format(suite_name, observer_name)
    suite = cocoex.Suite(suite_name, 'instances: 1', 'dimensions: 2 function_indices: 1')
    observer1 = cocoex.Observer(observer_name, 'result_folder: {}-1'.format(folder))
    observer2 = cocoex.Observer(observer_name, 'result_folder: {}-2'.format(folder))

    np.random.seed(12345)

    problem = suite[0]
    problem.observe_with(observer1)
    problem.observe_with(observer2)
    for _ in range(10):
        x = np.random.uniform(problem.lower_bounds, problem.upper_bounds, problem.dimension)
        problem(x)

    return observer1.result_folder, observer2.result_folder


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    if 11 < 3:
        # Experiments with two observers
        run_two_observers('bbob', 'bbob-old')
        run_two_observers('bbob-biobj', 'bbob-biobj')

    # Experiments with a single observer
    cocoex.known_suite_names.append(b"bbob-constrained")
    suite_options_1 = 'dimensions: 5,40 function_indices: 9-20'
    observer_options_1 = 'lin_target_precision: 10e6'
    settings = [
        dict(suite_name='bbob', suite_options=suite_options_1,
             observer_name='bbob', observer_options=observer_options_1,
             instance_order='def'),
        dict(suite_name='bbob', suite_options=suite_options_1,
             observer_name='bbob', observer_options=observer_options_1,
             instance_order='inst'),
        dict(suite_name='bbob', suite_options=suite_options_1,
             observer_name='bbob', observer_options=observer_options_1,
             instance_order='rand'),

        dict(suite_name='bbob', suite_options=suite_options_1,
             observer_name='bbob-old', observer_options=observer_options_1,
             instance_order='def'),
        dict(suite_name='bbob', suite_options=suite_options_1,
             observer_name='bbob-old', observer_options=observer_options_1,
             instance_order='inst'),

        dict(suite_name='bbob-constrained', suite_options=suite_options_1,
             observer_name='bbob-old', observer_options=observer_options_1,
             instance_order='def'),
        dict(suite_name='bbob-constrained', suite_options=suite_options_1,
             observer_name='bbob', observer_options=observer_options_1,
             instance_order='def'),
        dict(suite_name='bbob-mixint', suite_options=suite_options_1,
             observer_name='bbob', observer_options=observer_options_1,
             instance_order='def'),

        dict(suite_name='bbob-biobj', suite_options=suite_options_1,
             observer_name='bbob-biobj', observer_options=observer_options_1,
             instance_order='def'),
        dict(suite_name='bbob-biobj-mixint', suite_options=suite_options_1,
             observer_name='bbob-biobj', observer_options=observer_options_1,
             instance_order='def'),
    ]

    for setting in settings:
        folder = '{}_{}_{}'.format(setting['suite_name'], setting['observer_name'],
                                   setting['instance_order'])
        run_experiment(folder=folder, **setting)
