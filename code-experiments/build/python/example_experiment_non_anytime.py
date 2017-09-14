#!/usr/bin/env python
"""An example experiment for benchmarking non-anytime optimization algorithms with restarts.

Non-anytime algorithms are those whose parameters depend not only on the dimensionality of the problem, but also on the
budget of evaluations. Benchmarking such algorithms therefore entails running experiments on a test suite using various
(increasing) budgets. These budgets can/should be provided by the user. If the user does not specify a budget list, a
default budget list is used.

This code builds upon the example experiment for beginners and provides only simplistic progress information.

To apply the code to a different solver, add an interface to function `fmin`.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import cocoex  # experimentation module
from numpy.random import rand  # for randomised restarts

# for postprocessing
import cocopp
import os
import webbrowser

# for solvers
import scipy.optimize
import cma


def default_budget_list(max_budget=10, num=100):
    """Produces a budget list with at most `num` different increasing budgets within [1, `max_budget`] that are equally
    spaced in the logarithmic space.
    """
    from math import log10
    from numpy import unique, logspace
    return unique(logspace(0, log10(max_budget), num=num).astype(int))


def fmin(problem, x0, solver, budget):
    """Invokes `solver` on `problem` with initial solution `x0`.

    `solver` should evaluate the final/returned solution. Add here the interface to your solver.
    """
    if solver.__name__ == 'fmin' and solver.__globals__['__name__'] in ['cma', 'cma.evolution_strategy', 'cma.es']:
        return solver(problem, x0, 2, {'verbose': -9})  # set 'verbose': 1 for more output

    elif solver.__name__ == 'fmin' and solver.__globals__['__name__'] == 'scipy.optimize.optimize':
        return solver(problem, x0, xtol=1e-9, ftol=1e-9, maxfun=budget, disp=False)  # set disp=True for more output

    elif solver.__name__ == 'differential_evolution':
        pop_size_multiplier = 5  # pop_size = pop_size_multiplier * problem.dimension
        # a strange but correct way to set max_iter so that the budget is respected
        max_iter = max(1, int(budget / pop_size_multiplier / problem.dimension) - 1)
        bounds = [x for x in zip(problem.lower_bounds, problem.upper_bounds)]
        return solver(problem, bounds, popsize=pop_size_multiplier, maxiter=max_iter, strategy='best1bin',
                      tol=1e-9, polish=False, disp=False)  # set disp=True for more output

    else:
        return solver(problem, x0)


def main():
    ### input
    suite_name = "bbob"
    solver = scipy.optimize.fmin  # the Nelder-Mead downhill simplex method
    # solver = scipy.optimize.differential_evolution  # to use differential evolution instead
    # solver = cma.fmin  # to use CMA-ES instead
    algorithm_name = "Nelder-Mead"  # no spaces allowed
    output_folder = algorithm_name  # no spaces allowed

    # a list of increasing budgets (to be additionally multiplied by dimension)
    # gradually increase `max_budget` to 10, 1000, ... or replace with a user-defined list
    budget_multiplier_list = default_budget_list(max_budget=5)

    ### prepare (see http://numbbo.github.io/coco-doc/C/ for suite and observer parameters)
    suite = cocoex.Suite(suite_name, "", "")
    observer = cocoex.Observer(suite_name, "result_folder: {} algorithm_name: {}".format(output_folder, algorithm_name))

    ### go
    for index, _ in enumerate(suite):
        with ProblemNonAnytime(suite, observer, index) as problem:
            problem.print_progress()
            for budget in [problem.dimension * x for x in budget_multiplier_list]:  # iterate over budgets
                problem.reset()
                x0 = problem.initial_solution  # initial solution
                # apply restarts while neither the problem is solved nor the budget is exhausted
                while problem.evaluations < budget and not problem.final_target_hit:
                    fmin(problem, x0, solver, budget)
                    x0 = problem.random_solution  # a random initial solution for restarted algorithms
            problem.free()

    ### post-process data
    cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")


class ProblemNonAnytime(object):
    """The non-anytime problem class.

    Contains two problems - one observed and the other unobserved. It switches from unobserved to observed when
    p_unobserved.evaluations >= p_observed.evaluations.
    """
    def __init__(self, suite, observer, index):
        self.suite = suite
        self.p_unobserved = suite[index]
        self.p_observed = suite[index].add_observer(observer)
        self._p = self.p_observed
        self.evaluations = 0
        for key in ['dimension',
                    'id',
                    'index',
                    'info',
                    'initial_solution',
                    'lower_bounds',
                    'name',
                    'number_of_constraints',
                    'number_of_objectives',
                    'number_of_variables',
                    'upper_bounds']:
            setattr(self, key, getattr(self._p, key))

    def __call__(self, *arg, **args):
        if self.evaluations > self.p_observed.evaluations:
            raise ValueError("Evaluations {} are larger than observed evaluations {}".format(
                self.evaluations, self.p_observed.evaluations))
        if self.evaluations >= self.p_observed.evaluations:
            self._p = self.p_observed
        self.evaluations += 1
        return self._p(*arg, **args)

    def reset(self):
        self.evaluations = 0
        self._p = self.p_unobserved

    def free(self):
        self.p_observed.free()
        self.p_unobserved.free()

    def print_progress(self):
        if not self.index % (len(self.suite) / len(self.suite.dimensions)):
            print("\nd={}: ".format(self.dimension))
        print("{}{} ".format(self.id[self.id.index("_f")+1:self.id.index("_i")],
                             self.id[self.id.index("_i"):self.id.index("_d")]), end='', flush=True)
        if not (self.index + 1) % 10:
            print("")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()

    @property
    def final_target_hit(self):
        return self.p_observed.final_target_hit

    @property
    def best_observed_fvalue1(self):
        return self.p_observed.best_observed_fvalue1

    @property
    def random_solution(self):
        return self.lower_bounds + ((rand(self.dimension) + rand(self.dimension)) *
                                    (self.upper_bounds - self.lower_bounds) / 2)


if __name__ == '__main__':
    main()
