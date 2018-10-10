#!/usr/bin/env python
"""Code to test the bbob-mixint suite"""

from __future__ import division, print_function
import cocoex
import numpy as np
from numpy.random import rand
import scipy.optimize as opt
import cma

budget_multiplier = 1e5


def solve(solver_name, problem, x0, budget, de_pop_mult):
    if solver_name == "de":
        # a strange but correct way to set max_iter to respect the budget
        max_iter = max(1, int(budget / de_pop_mult / problem.dimension) - 1)
        de = opt.differential_evolution
        return de(problem,
                  [x for x in zip(problem.lower_bounds, problem.upper_bounds)],
                  popsize=de_pop_mult,
                  maxiter=max_iter,
                  strategy='best1bin',
                  tol=1e-9,
                  polish=False,
                  seed=224,
                  disp=False)  # set disp=True for more output
    elif solver_name == "cma-b-int":
        # CMA with bounds and integer variables
        return cma.fmin(problem, x0, 0.5, {'verbose': -9,
                                           'maxfevals': budget,
                                           'bounds': [problem.lower_bounds, problem.upper_bounds],
                                           'integer_variables': np.arange(
                                               problem.number_of_integer_variables)})
    elif solver_name == "cma-b-cont":
        # CMA with bounds and continuous variables
        return cma.fmin(problem, x0, 0.5, {'verbose': -9,
                                           'maxfevals': budget,
                                           'bounds': [problem.lower_bounds, problem.upper_bounds]})
    elif solver_name == "cma-cont":
        # CMA without bounds and with continuous variables
        return cma.fmin(problem, x0, 2, {'verbose': -9,
                                         'maxfevals': budget})


def run_experiment(suite_name, observer_name, solver_name):
    suite = cocoex.Suite(suite_name, "", "")
    observer = cocoex.Observer(observer_name, "result_folder: {}-{}".format(solver_name, suite_name))
    minimal_print = cocoex.utilities.MiniPrint()

    for problem in suite:
        problem.observe_with(observer)
        x0 = problem.initial_solution
        # population size and population sizde multiplier for DE
        de_pop_size = int(3 * np.power(np.log10(problem.dimension * budget_multiplier), 2))
        de_pop_mult = max(1, int(de_pop_size / problem.dimension))
        # apply restarts while neither the problem is solved nor the budget is exhausted
        while (problem.evaluations < problem.dimension * budget_multiplier
               and not problem.final_target_hit):
            remaining_budget = problem.dimension * budget_multiplier - problem.evaluations
            solve(solver_name, problem, x0, remaining_budget, de_pop_mult)
            x0 = problem.lower_bounds + ((rand(problem.dimension) + rand(problem.dimension)) *
                                         (problem.upper_bounds - problem.lower_bounds) / 2)
        minimal_print(problem, final=problem.index == len(suite) - 1)


def run_all():
    observer_name = "bbob"
    for suite_name in ["bbob-mixint-1", "bbob-mixint-2"]:
        for solver_name in ["cma-b-int", "cma-b-cont", "cma-cont", "de"]:
            try:
                run_experiment(suite_name, observer_name, solver_name)
            except ValueError as e:
                print(e)


if __name__ == '__main__':
    run_all()
