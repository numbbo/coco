#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import doctest
import numpy as np
import cocoex as ex
from cocoex import Suite
from cocoex.utilities import about_equal
from cocoex import known_suite_names
import example_experiment

default_testcases = ["bbob2009_testcases.txt"]

def read_test_vectors(fd):
    """
    Read the number of test vectors, followed by the 40D test vectors
    from ${fd}. Return a list of numpy arrays containing the test vectors.
    """
    number_of_test_vectors = int(fd.readline().rstrip())
    ## Preallocate the testvectors list
    test_vectors = number_of_test_vectors * [None]
    for i in range(number_of_test_vectors):
        line = fd.readline().rstrip()
        test_vectors[i] = np.fromstring(line, dtype=float, sep=" ")
    return test_vectors

def process_test_cases(fd, suite_name, test_vectors):
    """
    Read test cases for benchmark suite ${suite_name} from ${fd} and evaluate them.
    """
    number_of_testcases = 0
    number_of_failures = 0
    previous_problem_index = None
    suite = Suite(suite_name, "instances:1-15", "")
    print("Testing suite", suite_name)
    for test_case in fd:
        number_of_testcases += 1

        ## A test case is a 4-tuple (deprecated_problem_index, problem_index, test_vector_id,
        ## expected_y) separated by a tab.
        deprecated_problem_index, problem_index, test_vector_id, expected_y = test_case.split()
        ## Do type conversion. Python gurus probably know an elegant
        ## one line solution...
        problem_index = int(problem_index)
        test_vector_id = int(test_vector_id)
        expected_y = float(expected_y)

        ## We cache the problem instances because creating an instance
        ## can be expensive depending on the transformation.
        if problem_index != previous_problem_index:
            problem = suite.get_problem(int(problem_index))
            previous_problem_index = problem_index
        test_vector = test_vectors[test_vector_id]
        y = problem(test_vector[:problem.number_of_variables])
        if not about_equal(y, expected_y, 4e-6):
            number_of_failures += 1
            print(f"{problem.id}:{test_vector_id} FAILED expected={expected_y} observed={y}")
            #elif number_of_failures == 100:
            #    print("... further failed tests suppressed ...")
    print("%i of %i tests passed (failure rate %.2f%%)" % (number_of_testcases - number_of_failures, number_of_testcases, (100.0 * number_of_failures) / number_of_testcases))
    if number_of_failures > 0:
        sys.exit(-1)

def process_testfile(testfile):
    with open(testfile, "r") as fd:
        test_suite = fd.readline().rstrip()
        test_vectors = read_test_vectors(fd)
        process_test_cases(fd, test_suite, test_vectors)

def testmod(module):
    """`doctest`s `testmod` method with `raise_on_error=True` setting"""
    print("  doctest of %s" % str(module))
    doctest.testmod(module,  # optionflags=doctest.ELLIPSIS,
                    raise_on_error=True)

def best_parameter(f):
    f._best_parameter('print')
    with open('._bbob_problem_best_parameter.txt', 'rt') as file_:
        return [float(s) for s in file_.read().split()]

def run_constrained_suite_test():
    from collections import defaultdict
    try:
        suite = Suite('bbob-constrained', '', '')
    except NameError:
        return
    counts = defaultdict(int)
    for f in suite:
        counts[-5] += np.any(f.initial_solution < -5)
        counts[5] += np.any(f.initial_solution > 5)
        counts['c'] += np.any(f.constraint(f.initial_solution) > 0)
        counts['b'] += np.any(f.constraint(best_parameter(f)) > 1e-11)  # mac: 6.8361219664552603e-12 is the largest value
    assert sum(counts.values()) == 0

def run_doctests():
    """Run doctests on "all" modules.

    To include this in a unittest environment,
    see https://docs.python.org/2/library/doctest.html#unittest-api
    """
    interface = ex.interface if hasattr(ex, 'interface') else ex._interface
    testmod(ex)
    if not sys.version.startswith('3'):
        print("  CAVEAT: doctest OF cocoex.interface IS, FOR SOME REASON, " +
              "INEFFECTIVE IN PYTHON 2 ")
    testmod(interface)
    testmod(example_experiment)


def _clean_up(folder, start_matches, protected):
    """permanently remove entries in `folder` which begin with any of
    `start_matches`, where `""` matches any string, and which are not in
    `protected`.

    CAVEAT: use with care, as with `"", ""` as second and third arguments
    this deletes all folder entries like `rm *` does. """
    if not os.path.isdir(folder):
        return
    if not protected and "" in start_matches:
        raise ValueError(
            '_clean_up(folder, [..., "", ...], []) is not permitted, resembles "rm *"')
    for d in os.listdir(folder):
        if d not in protected:
            for name in start_matches:
                if d.startswith(name):
                    shutil.rmtree(os.path.join(folder, d))
                    break


def main(args):
    list_before = os.listdir('exdata') if os.path.isdir('exdata') else []
    print('Running doctests...'), sys.stdout.flush()
    run_doctests()
    print('doctests done.\nRunning example_experiment:'), sys.stdout.flush()
    example_experiment.main()
    if "bbob-constrained" in known_suite_names:
        run_constrained_suite_test()
    for arg in args if args else default_testcases:
        if arg is None or arg == 'None':
            break
        process_testfile(arg) if args or os.path.isfile(arg) else None
    _clean_up('exdata', ['random_search_on_bbob', 'doctest', 'default'], list_before)

if __name__ == '__main__':
    main(sys.argv[1:])
