#!/usr/bin/env python
"""A short and simple example experiment with restarts.

The script is fully functional but also emphasises on readability. It
features restarts, timings and recording termination conditions.

To benchmark a different solver, `fmin` must be re-assigned and another
`elif` block added around line 70 to account for the solver-specific
call.

When calling the script, variables can be re-assigned via a
``name=value`` argument white spaces, where ``value`` is interpreted as
a single python literal. Additionally, ``batch`` is recognized as
argument defining the `current_batch` number and the number of
`batches`, like ``batch=2/8`` runs batch 2 of 8.

Examples, preceeded by "python" in an OS shell and by "run" in an IPython
shell::

    example_experiment2.py budget_multiplier=3  # times dimension

    example_experiment2.py budget_multiplier=1e4 cocopp=None

    example_experiment2.py budget_multiplier=1000 batch=1/16

Details: ``batch=9/8`` is equivalent to ``batch=1/8``. The first number
is taken modulo to the second.

"""
from __future__ import division, print_function, unicode_literals
__author__ = "Nikolaus Hansen and ..."
import sys
import time  # output some timings per evaluation
from collections import defaultdict
import os, webbrowser  # to show post-processed results in the browser
import numpy as np  # for np.median
import cocoex  # experimentation module
import cocopp  # post-processing module, comment out if post-processing is not desired

### solver imports (add other imports if necessary)
import scipy.optimize  # to define the solver to be benchmarked
# import cma

### input (to be modified if necessary/desired)
# fmin = scipy.optimize.fmin
fmin = scipy.optimize.fmin_slsqp
# fmin = scipy.optimize.fmin_cobyla
# fmin = cocoex.solvers.random_search
# fmin = cma.fmin2

suite_name = "bbob"  # see cocoex.known_suite_names
budget_multiplier = 2  # times dimension, increase to 10, 100, ...
omit_last_dimension = False

batches = 1  # number of batches, batch=3/32 works to set both, current_batch and batches
current_batch = 1  # only current_batch modulo batches is relevant

### possibly modify/overwrite above input parameters from input args
if __name__ == "__main__":
    input_params = cocoex.utilities.args_to_dict(
        sys.argv[1:], globals(), {'batch': 'current_batch/batches'}, print=print)
    globals().update(input_params)  # (re-)assign variables

# overwrites folder input parameter, comment out if desired otherwise
output_folder = '%s_of_%s_%dD_on_%s' % (fmin.__name__, fmin.__module__,
                                        int(budget_multiplier), suite_name)

### prepare
suite = cocoex.Suite(suite_name, "", "")
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()
stoppings = defaultdict(list)  # dict of lists, key is the problem index
timings = defaultdict(list)  # key is the dimension

### go
print('*** benchmarking %s from %s on suite %s ***'
      % (fmin.__name__, fmin.__module__, suite_name))
time0 = time.time()
for problem in suite:  # this loop may take several minutes or hours or days...
    if omit_last_dimension and problem.dimension == max(suite.dimensions):
        break
    if problem.index % batches != current_batch % batches:
        continue
    if not len(timings[problem.dimension]) and len(timings) > 1:
        print("\n    done in %.1e seconds/evaluations"
              % np.median(timings[sorted(timings)[-2]]), end='')
    problem.observe_with(observer)  # generate the data for cocopp post-processing
    maxevals = problem.dimension * budget_multiplier  # just giving a short name
    time1 = time.time()

    # apply restarts
    while max((problem.evaluations, problem.evaluations_constraints)) < maxevals and not problem.final_target_hit:
        x0 = problem.initial_solution_proposal()  # give different proposals, all zeros in first call
        # here we assume that `fmin` evaluates the final/returned solution:
        if fmin is scipy.optimize.fmin:
            output = fmin(problem, x0, maxfun=maxevals, disp=False, full_output=True)
            stoppings[problem.index].append(output[4])
        elif fmin is scipy.optimize.fmin_slsqp:
            output = fmin(problem, x0, iter=int(budget_multiplier+1),  # very approximate way to respect budget
                          full_output=True, iprint = -1)
            # print(problem.dimension, problem.evaluations)
            stoppings[problem.index].append(output[3:])
        elif fmin is cocoex.solvers.random_search:
            fmin(problem, problem.dimension * [-5], problem.dimension * [5],
                 maxevals)
        elif fmin.__name__ == 'fmin2' and fmin.__module__ == 'cma':  # cma.fmin2:
            xopt, es = fmin(problem, problem.initial_solution_proposal(), 2,
                            {'maxfevals':maxevals, 'verbose':-9}, restarts=7)
            stoppings[problem.index].append(es.stop())
        elif fmin is scipy.optimize.fmin_cobyla:
            fmin(problem, x0, lambda x: -problem.constraint(x), maxfun=maxevals,
                   disp=0, rhoend=1e-9)
        # add another solver here

    timings[problem.dimension].append((time.time() - time1) / problem.evaluations
                                      if problem.evaluations else 0)
    minimal_print(problem, restarted=any(x0 != problem.initial_solution),
                  final=problem.index == len(suite) - 1)
    with open(output_folder + '_stopping_conditions.out', 'wt') as file_:
        file_.write("# code to read in these data:\n"
                    "# import ast\n"
                    "# with open('%s_stopping_conditions.out', 'rt') as file_:\n"
                    "#     stoppings = ast.literal_eval(file_.read())\n"
                    % output_folder)
        file_.write(repr(stoppings))

### print timings and final message
print("\n  dimension  median seconds/evaluations")
print("  -------------------------------------")
for dimension in sorted(timings):
    print("    %3d       %.1e" % (dimension, np.median(timings[dimension])))
print("  -------------------------------------")
if batches > 1:
    print("*** Batch %d of %d batches finished in %s."
          " Make sure to run *all* batches (via current_batch or batch=#/#) ***"
          % (current_batch, batches, cocoex.utilities.ascetime(time.time() - time0)))
else:
    print("*** Full experiment done in %s ***"
          % cocoex.utilities.ascetime(time.time() - time0))

### post-process data
if batches == 1 and 'cocopp' in globals() and cocopp not in (None, 'None'):
    cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")