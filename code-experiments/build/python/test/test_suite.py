import pytest
import numpy as np

from cocoex import Suite
from pathlib import Path

TEST_DIR = Path(__file__).parent


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


def test_bbob2009():
    with open(TEST_DIR / "bbob2009_testcases.txt") as fd:
        test_suite = fd.readline().rstrip()
        test_vectors = read_test_vectors(fd)
      
        suite = Suite(test_suite, "instances:1-15", "")
        previous_problem_index = -1
        for test_case in fd:
            ## A test case is a 4-tuple (deprecated_problem_index, problem_index, test_vector_id,
            ## expected_y) separated by a tab.
            deprecated_problem_index, problem_index, test_vector_id, expected_y = test_case.split()

            ## We cache the problem instances because creating an instance
            ## can be expensive depending on the transformation.
            if problem_index != previous_problem_index:
                problem = suite.get_problem(int(problem_index))
                previous_problem_index = problem_index
            test_vector = test_vectors[int(test_vector_id)]
            y = problem(test_vector[:problem.number_of_variables])
            assert y == pytest.approx(float(expected_y))
