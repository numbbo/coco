#!/usr/bin/env python
"""An example experiment for benchmarking non-anytime optimization algorithms
with restarts.

Non-anytime algorithms are those whose parameters depend not only on the
dimensionality of the problem, but also on the budget of evaluations.
Benchmarking such algorithms therefore entails running experiments on a test
suite using various (increasing) budgets. These budgets can/should be provided
by the user. If the user does not specify a budget list, a default budget list
is used.

This code builds upon the example experiment for beginners and provides only
simplistic progress information.

To apply the code to a different solver, add an interface to function `fmin`.
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
import cocoex  # experimentation module
import numpy as np

# for postprocessing
import cocopp
import os
import webbrowser

# for solvers
import scipy.optimize
import cma


def default_budget_list(max_budget=10, num=50):
    """Produces a budget list with at most `num` different increasing budgets
    within [1, `max_budget`] that are equally spaced in the logarithmic space.
    """
    return np.unique(np.logspace(0, np.log10(max_budget), num=num).astype(int))


def fmin(problem, x0, solver, budget):
    """Invokes `solver` on `problem` with `budget` and initial solution `x0`.
    Returns the final solution.

    Add here the interface to your solver.
    """
    if solver.__name__ == 'fmin':

        if solver.__globals__['__name__'] in ['cma', 'cma.evolution_strategy',
                                              'cma.es']:
            # set verbose=1 for more output
            result = solver(problem, x0, 2, options=dict(maxfevals=budget,
                                                         verbose=-9))
            return result[0]

        elif solver.__globals__['__name__'] == 'scipy.optimize.optimize':
            result = solver(problem, x0, xtol=1e-9, ftol=1e-9, maxfun=budget,
                            disp=False)  # set disp=True for more output
            return result

        else:
            return solver(problem, x0)

    elif solver.__name__ == 'differential_evolution':
        pop_size_multiplier = 5  # pop_size=pop_size_multiplier*dimension
        # a strange but correct way to set max_iter to respect the budget
        max_iter = max(1, int(
            budget / pop_size_multiplier / problem.dimension) - 1)
        bounds = [x for x in zip(problem.lower_bounds, problem.upper_bounds)]
        result = solver(problem, bounds, popsize=pop_size_multiplier,
                        maxiter=max_iter, strategy='best1bin',
                        tol=1e-9, polish=False,
                        disp=False)  # set disp=True for more output
        return result.x

    else:
        return solver(problem, x0)


def main():
    ### input
    suite_name = "bbob"
    solver = scipy.optimize.fmin  # the Nelder-Mead downhill simplex method
    # solver = scipy.optimize.differential_evolution  # to use DE instead
    # solver = cma.fmin  # to use CMA-ES instead
    algorithm_name = "Nelder-Mead"  # no spaces allowed
    output_folder = algorithm_name  # no spaces allowed

    # a list of increasing budgets to be multiplied by dimension
    # gradually increase `max_budget` to 10, 100, ...
    # or replace with a user-defined list
    budget_multiplier_list = default_budget_list(max_budget=5)
    print("Benchmarking with budgets: ", end="")
    print(", ".join(str(b) for b in budget_multiplier_list), end="")
    print(" (* dimension)")

    ### prepare (see http://numbbo.github.io/coco-doc/C/ for parameters)
    suite = cocoex.Suite(suite_name, "", "")
    observer = cocoex.Observer(suite_name,
                               "result_folder: {0} algorithm_name: {1}".format(
                                   output_folder, algorithm_name))

    ### go
    for index in range(len(suite)):
        with ProblemNonAnytime(suite, observer, index) as problem:
            problem.print_progress()
            x0 = problem.initial_solution  # initial solution
            for budget in [b * problem.dimension for b in
                           budget_multiplier_list]:  # iterate over budgets
                x = fmin(problem, x0, solver, budget)
                problem.delivered(x)
                if problem.final_target_hit:
                    break

    ### post-process data
    cocopp.main(observer.result_folder)  # re-run folders look like "...-001"
    webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")


class ProblemNonAnytime(object):
    """The non-anytime problem class.

    Serves to benchmark a "budgeted" algorithm whose behavior decisively
    depends on a budget input parameter.

    Usage::

        # given: suite and observer instances, budget_list, fmin
        for index in range(len(suite)):
            with ProblemNonAnytime(suite, observer, index) as problem:
                x0 = problem.initial_solution  # or whatever fmin needs as input
                for budget in sorted(budget_list):
                    x = fmin(problem, x0, budget)  # minimize
                    problem.delivered(x)  # prepares for next budget also
                    if problem.final_target_hit:
                        break

    Details: This class maintains two problems - one observed and the
    other unobserved. It switches from unobserved to observed when
    ``p_unobserved.evaluations >= p_observed.evaluations``.

    """
    inherited_constant_attributes = [
            'dimension',
            'id',
            'index',
            'info',
            'initial_solution',
            'lower_bounds',
            'name',
            'number_of_constraints',
            'number_of_objectives',
            'number_of_variables',
            'upper_bounds']

    def __init__(self, suite, observer, index):
        self.suite = suite
        self.p_unobserved = suite[index]
        self.p_observed = suite[index].add_observer(observer)
        self._p = self.p_observed
        self.evaluations = 0
        self._number_of_delivery_evaluations = 0  # just FTR
        for key in ProblemNonAnytime.inherited_constant_attributes:
            setattr(self, key, getattr(self._p, key))

    def __call__(self, x, *args, **kwargs):
        if self.evaluations > self.p_observed.evaluations:
            raise ValueError(
                "Evaluations {0} are larger than observed evaluations {1}"
                "".format(self.evaluations, self.p_observed.evaluations))
        if self.evaluations >= self.p_observed.evaluations:
            self._p = self.p_observed
        self.evaluations += 1
        return self._p(x, *args, **kwargs)

    def delivered(self, x, *args, **kwargs):
        """to be called with the solution returned by the solver.

        The method records the delivered solution (if necessary) and
        prepares the problem to be run on the next budget by calling
        `reset`.
        """
        if self.evaluations < self.p_observed.evaluations:
            raise ValueError(
                "Delivered solutions should come from increasing budgets,\n"
                "however the current budget = {0} is smaller than the "
                "delivery budget = {1}".format(
                    self.p_observed.evaluations, self.evaluations))
        # assuming that "same fitness" means the solution was already evaluated
        if self.p_unobserved(x, *args, **kwargs) != self.best_observed_fvalue1:
            # ideally we would decrease the observed evaluation counter,
            # but instead we now increase it only one time step later in
            # __call__. That is, in effect, we exchange the next solution
            # to be observed with this delivered solution `x`.
            self.p_observed(x, *args, **kwargs)
            self._number_of_delivery_evaluations += 1  # just FTR
        self.reset()

    def reset(self):
        """prepare the problem to be run on the next budget"""
        self.evaluations = 0
        self._p = self.p_unobserved

    def free(self):
        self.p_observed.free()
        self.p_unobserved.free()

    def print_progress(self):
        if not self.index % (len(self.suite) / len(self.suite.dimensions)):
            print("\nd={0}: ".format(self.dimension))
        print("{0}{1} ".format(
            self.id[self.id.index("_f") + 1:self.id.index("_i")],
            self.id[self.id.index("_i"):self.id.index("_d")]), end="",
              flush=True)
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
        return self.lower_bounds + (
            (np.random.rand(self.dimension) + np.random.rand(self.dimension))
            * (self.upper_bounds - self.lower_bounds) / 2)


if __name__ == '__main__':
    main()
