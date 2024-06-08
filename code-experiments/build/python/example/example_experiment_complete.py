#!/usr/bin/env python
"""A short yet complete example experiment script with restarts and batching.

Arguments
---------
This script must be called with 1-3 arguments:

    budget_multiplier [number_of_batches batch_to_execute]

``budget_multiplier`` times dimension is the budget within which problem
instances are run repeatedly as long as too few successes are observed.

``batch_to_execute`` can only be omitted when ``number_of_batches == 1``.
When ``number_of_batches > 1``, the script must be executed repeatedly
(e.g., in parallel) with the different values for ``batch_to_execute =
0..number_of_batches-1`` to get data for the full experiment.

Usage
-----
To apply the code to a different solver/algorithm, `fmin` must be
re-assigned or re-defined accordingly, and the below code must be edited at
the two places marked with "### input" around lines 40 and 80.

See also: https://numbbo.it/getting-started/experiment-python.html
"""

__author__ = "Nikolaus Hansen"
__copyright__ = "public domain"

import mkl_bugfix  # set *_NUM_THREADS=1, requires mkl_bugfix.py file from build/python/example
import collections
import time
import cocoex  # experimentation module
scipy = cocoex.utilities.forgiving_import('scipy')  # solvers to benchmark
cma = cocoex.utilities.forgiving_import('cma')  # solvers to benchmark

### input: define suite and solver (see also "input" below where fmin is called)
suite_name = "bbob"
# fmin = scipy.optimize.fmin  # optimizer to be benchmarked
fmin = cocoex.solvers.random_search
# fmin = cma.fmin2
# fmin = cma.fmin_lq_surr2
# fmin = scipy.optimize.fmin_slsqp

### reading in parameters
if __name__ == '__main__':
    import sys
    try:
        budget_multiplier = float(sys.argv[1])
        number_of_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        batch_to_execute = int(sys.argv[3]) if len(sys.argv) > 3 else None
    except Exception as e:
        print("Exception {} with calling arguments {}\n\n".format(e, sys.argv)
              + __doc__)
        raise

### prepare
suite = cocoex.Suite(suite_name, "", "")  # see https://numbbo.github.io/coco-doc/C/#suite-parameters
output_folder = '{}_of_{}_{}D_on_{}{}'.format(
        fmin.__name__, fmin.__module__ or '', int(budget_multiplier+0.499), suite_name,
        ('_batch{:0' + str(len(str(number_of_batches-1))) + '}of{}').format(
            batch_to_execute, number_of_batches) if number_of_batches > 1 else '')
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)  # see https://numbbo.github.io/coco-doc/C/#observer-parameters
repeater = cocoex.ExperimentRepeater(budget_multiplier)  # x dimension
batcher = cocoex.BatchScheduler(number_of_batches, batch_to_execute)
minimal_print = cocoex.utilities.MiniPrint()
timings = collections.defaultdict(list)  # key is the dimension
final_conditions = collections.defaultdict(list)  # key is (id_fun, dimension, id_inst)
cocoex.utilities.write_setting(locals(), [observer.result_folder, 'parameters.pydat'])

### go
time0 = time.time()
while not repeater.done():  # while budget is left and successes are few
    for problem in suite:  # loop takes 2-3 minutes x budget_multiplier
        if not batcher.is_in_batch(problem) or repeater.done(problem):
            continue  # skip problem and bypass repeater.track
        problem.observe_with(observer)  # generate data for cocopp
        time1 = time.time()
        problem(problem.dimension * [0])  # for better comparability

        ### input: implement/amend the next few lines for another fmin
        if fmin == 'my solver':
            res = fmin(problem, repeater.initial_solution_proposal(problem),
                       )
            xopt = res[0]
            final_condition = res[1]  # store this output to scrutinize later
        elif fmin == cocoex.solvers.random_search:  # needs budget_multiplier as input
            xopt = fmin(problem, problem.dimension * [-5], problem.dimension * [5],
                        problem.dimension * budget_multiplier)
            final_condition = None
        elif fmin == scipy.optimize.fmin:
            res = fmin(problem, repeater.initial_solution_proposal(problem),
                    disp=False, full_output=True)
            xopt = res[0]
            final_condition = res[4]  # store this output to scrutinize later
        elif fmin == scipy.optimize.fmin_slsqp:
            res = fmin(problem, repeater.initial_solution_proposal(problem),
                          acc=1e-11, full_output=True, iprint = -1)
            xopt = res[0]
            final_condition = res[3:]
        elif fmin == scipy.optimize.fmin_cobyla:  # on bbob-constrained suite
            xopt = fmin(problem, repeater.initial_solution_proposal(problem),
                 lambda x: -problem.constraint(x),
                 maxfun=max((1000, problem.dimension * budget_multiplier)),
                 disp=0, rhoend=1e-9)
            final_condition = None
        elif fmin in (cma.fmin2, cma.fmin_lq_surr2):
            if fmin == cma.fmin_lq_surr2 and problem.dimension == 40:
                continue  # takes prohibitively long!?
            options = {'maxfevals': problem.dimension * budget_multiplier,
                       'termination_callback': lambda es: problem.final_target_hit,
                       'conditioncov_alleviate': 2 * [False] if fmin == cma.fmin_lq_surr2 else None,
                       'verbose': -9 }
            xopt, es = fmin(problem, problem.initial_solution_proposal, 2,
                            options, restarts=9)
            final_condition = es.stop()
        else:
            raise ValueError('case for fmin={} not found'.format(fmin))

        problem(xopt)  # make sure the returned solution is evaluated

        timings[problem.dimension].append((time.time() - time1) / problem.evaluations)
        repeater.track(problem)  # track evaluations and final_target_hit
        minimal_print(problem)  # show progress
        final_conditions[problem.id_triple].append(repr(final_condition))
        with open(observer.result_folder + '/final_conditions.pydict', 'wt') as file_:
            file_.write(str(dict(final_conditions)).replace('],', '],\n'))

### final messaging
print("\nTiming summary:\n"
      "  dimension  median time [seconds/evaluation]\n"
      "  -------------------------------------")
for dimension in sorted(timings):
    ts = sorted(timings[dimension])
    print("    {:3}       {:.1e}".format(dimension, (ts[len(ts)//2] + ts[-1-len(ts)//2]) / 2))
print("  -------------------------------------")

if number_of_batches > 1:
    print("\n*** Batch {} of {} batches finished in {}."
          " Make sure to run *all* batches (0..{}) ***".format(
          batch_to_execute, number_of_batches,
          cocoex.utilities.ascetime(time.time() - time0), number_of_batches - 1))
else:
    print("\n*** Full experiment done in %s ***"
          % cocoex.utilities.ascetime(time.time() - time0))
print("    Data written into {}".format(observer.result_folder))

### post-process data
if number_of_batches == 1:
    print("    Postprocess with 'python cocopp {} [...]'".format(observer.result_folder))
    import cocopp  # post-processing module
    dsl = cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
