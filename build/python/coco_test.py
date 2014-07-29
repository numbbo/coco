#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from coco import about_equal, Problem
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

def process_test_cases(fd, suit_name, test_vectors):
    """
    Read test cases for benchmark suit ${suit_name} from ${fd} and evaluate them.
    """
    number_of_testcases = 0
    number_of_failures = 0
    previous_function_id = None
    for test_case in fd:
        number_of_testcases += 1

        ## A test case is a triple (function_id, test_vector_id,
        ## expected_y) separated by a single space. 
        function_id, test_vector_id, expected_y = test_case.split(" ")
        ## Do type conversion. Python gurus probably know an elegant
        ## one line solution...
        function_id = int(function_id)
        test_vector_id = int(test_vector_id)
        expected_y = float(expected_y)

        ## We cache the problem instances because creating an instance
        ## can be expensive depending on the transformation.
        if function_id != previous_function_id:
            problem = Problem(suit_name, function_id)
            previous_function_id = function_id
        test_vector = test_vectors[test_vector_id]
        y = problem(test_vector[:problem.number_of_variables()])
        if not about_equal(y, expected_y):
            number_of_failures += 1
            if number_of_failures < 100:
                print("%8i %8i FAILED expected=%.8e observed=%.8e" % (function_id, test_vector_id, expected_y, y))
            elif number_of_failures == 100:
                print("... further failed tests suppressed ...")
    print("%i of %i tests passed (failure rate %.2f%%)" % (number_of_testcases - number_of_failures, number_of_testcases, (100.0 * number_of_failures) / number_of_testcases))
    if number_of_failures > 0: 
        sys.exit(-1)

def process_testfile(testfile):
    with open(testfile, "r") as fd:
        test_suit = fd.readline().rstrip()
        test_vectors = read_test_vectors(fd)
        process_test_cases(fd, test_suit, test_vectors)

def main(args):
    for arg in args:
        process_testfile(arg)

if __name__ == '__main__':
    main(sys.argv[1:])
