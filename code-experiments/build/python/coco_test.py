#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import doctest
import cocoex as ex
from cocoex import Suite
from cocoex.utilities import about_equal
import numpy as np
import sys

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
    suite = Suite(suite_name, "", "")
    print("Testing suite", suite_name)
    for test_case in fd:
        number_of_testcases += 1

        ## A test case is a 4-tuple (deprecated_problem_index, problem_index, test_vector_id,
        ## expected_y) separated by a tab. 
        deprecated_problem_index, problem_index, test_vector_id, expected_y = test_case.split("\t")
        ## Do type conversion. Python gurus probably know an elegant
        ## one line solution...
        deprecated_problem_index = int(deprecated_problem_index)
        test_vector_id = int(test_vector_id)
        expected_y = float(expected_y)

        ## We cache the problem instances because creating an instance
        ## can be expensive depending on the transformation.
        if deprecated_problem_index != previous_problem_index:
            problem = suite.get_problem(int(problem_index))
            previous_problem_index = deprecated_problem_index
        test_vector = test_vectors[test_vector_id]
        y = problem(test_vector[:problem.number_of_variables])
        if not about_equal(y, expected_y):
            number_of_failures += 1
            if number_of_failures < 100:
                print("%8i %8i FAILED expected=%.8e observed=%.8e" % (deprecated_problem_index, test_vector_id, expected_y, y))
            elif number_of_failures == 100:
                print("... further failed tests suppressed ...")
    print("%i of %i tests passed (failure rate %.2f%%)" % (number_of_testcases - number_of_failures, number_of_testcases, (100.0 * number_of_failures) / number_of_testcases))
    if number_of_failures > 0: 
        sys.exit(-1)

def process_testfile(testfile):
    with open(testfile, "r") as fd:
        test_suite = fd.readline().rstrip()
        test_vectors = read_test_vectors(fd)
        process_test_cases(fd, test_suite, test_vectors)

def testmod(module):
    doctest.testmod(module, optionflags=doctest.ELLIPSIS, raise_on_error=True)
    
def main(args):
    for arg in args:
        process_testfile(arg)

if __name__ == '__main__':
    interface = ex.interface if hasattr(ex, 'interface') else ex._interface
    testmod(interface)
    import example_experiment
    testmod(example_experiment)
    example_experiment.main()
    main(sys.argv[1:])
