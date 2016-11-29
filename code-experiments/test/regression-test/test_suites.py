#!/usr/bin/env python
"""test all known suites with data from 'data' folder, where
the optional input parameter is the number of data per
problem visible in the filename, by default 10.
"""
from __future__ import division, print_function
from ast import literal_eval  # safe evaluation/execution of Python code
import os, sys
import time
import numpy as np
import cocoex


def _is_equal(x, y):
    """return scalar of vector"""
    x, y = np.asarray(x), np.asarray(y)
    return (np.abs(x - y) < 1e-11) + (y * (1 - 1e-11) < x) * (x < y * (1 + 1e-11)) > 0


def is_equal(x, y):
    try:
        assert len(x) == len(y)
    except TypeError:
        """scalar case"""
        return _is_equal(x, y)
    else:
        return np.all(_is_equal(x, y))


def regression_test_a_suite(suite_name, filename):
    """filename contains previously generated test data to compare against
    """
    verbose = 1
    xfc_dict = literal_eval(open(filename).read())
    if verbose:
        print("using file %s with %d test cases " % (filename, len(xfc_dict)), end="")
        sys.stdout.flush()
        t0 = time.clock()
    suite = cocoex.Suite(suite_name, "", "")
    for key in xfc_dict:
        f, x = suite[key[0]], key[1]
        try:
            assert is_equal(f(x), xfc_dict[key][0])
        except AssertionError:
            print(f.name, key, xfc_dict[key], f(x))
            raise
        if f.number_of_constraints > 0:
            try:
                assert is_equal(f.constraint(x), xfc_dict[key][1])
            except AssertionError:
                print(f.name, key, xfc_dict[key], f.constraint(x))
                raise
    if verbose:
        print("done in %.1fs" % (time.clock() - t0))

if __name__ == "__main__":
    try:
        ndata = int(sys.argv[1])
    except IndexError:
        ndata = 10
    except ValueError:
        print(__doc__)
    try:
        ndata
    except:
        pass
    else:
        for name in cocoex.known_suite_names:
            regression_test_a_suite(name,
                os.path.join("data",
                             "regression_test_%ddata_for_suite_" % ndata + name + ".py"))
