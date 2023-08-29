
from __future__ import division, print_function
from ast import literal_eval  # safe evaluation/execution of Python code
import os, sys
import time
import numpy as np
import re
import pickle
from bbob_noisy_test_seeds import read_seed_file

best_values_legacy_path = "code-experiments/test/regression-test/regression-test-bbob-noisy/data_legacy/bbob_noisy_best_values.json"
best_values_path = "code-experiments/test/regression-test/regression-test-bbob-noisy/data/bbob_noisy_best_values.txt"


seeds_path = "code-experiments/test/regression-test/regression-test-bbob-noisy/data/bbob_noisy_seeds.txt"
seeds_legacy_path = "code-experiments/test/regression-test/regression-test-bbob-noisy/data_legacy/bbob_noisy_seeds.json"

seeds = read_seed_file(seeds_path)
with open(seeds_legacy_path, "rb") as file_:
    seeds_legacy = pickle.load(file_, encoding='latin1')


def _is_equal(x, y):
    """return scalar or vector, where `x` and `y` can be a scalar
    or list/array_type
    """
    x, y = np.asarray(x), np.asarray(y)
    same_sign = x * y > 0
    ax, ay = np.abs(x), np.abs(y)
    lgx, lgy = np.log10(ax), np.log10(ay)
    return ((np.abs(x - y) < 1e-9) +  # "+" means in effect "or"
            same_sign * (np.abs(x - y) / (ax + ay) < 2e-9) +  # min(ax, ay) would be better?
            same_sign * (ax > 1e21) * (ay > 1e21)  # *  # because coco.h defines INFINITY possibly as 1e22
           ) # (np.abs(lgx - lgy) / (lgx + lgy) < 0.7) > 0)  # probably not very useful 

def is_equal(x, y):
    try:
        assert len(x) == len(y)
    except TypeError:
        """scalar case"""
        return _is_equal(x, y)
    else:
        return np.all(_is_equal(x, y))


def encode_problem_index(function_info_string):

    target_fval = re.findall(".*optimal_f_value:\s*(-?\d*\.\d*)", function_info_string)[0]; target_fval = float(target_fval)
    function_suite_id = function_info_string.split(",")[0]    
    return function_suite_id, float(target_fval)



def parse_best_values_output_file(best_values_string):
    best_values_string = fr"{best_values_string}".split("\n")
    best_values_string = [best_value_string for best_value_string in best_values_string if (best_value_string and not re.match("([\{\}]|.*x.*)", best_value_string))]
    best_values_string_idxs = [index for index, best_value_string in enumerate(best_values_string) if re.match(".*optimal_f_value:.*", best_value_string)]
    best_values_output_dictionary = dict()
    for j, idx in enumerate(best_values_string_idxs):
        function_output_dictionary = dict()
        idx1 = None
        
        function_info_string = best_values_string[idx]        
        function_suite_id, target_fval = encode_problem_index(function_info_string)
        if j!= len(best_values_string_idxs) - 1:
            idx1 = best_values_string_idxs[j + 1]
        xopt_arr = np.array([float(re.sub("[\\t,]", "",  x)) for x in best_values_string[idx + 1: idx1]])
        function_output_dictionary["xopt"] = xopt_arr
        function_output_dictionary["ftarget"] = target_fval

        best_values_output_dictionary[function_suite_id] = function_output_dictionary

    return best_values_output_dictionary
    
def regression_test_best_values(best_values_dictionary, best_values_legacy_dictionary):
    global seeds_legacy
    global seeds
    assert (len(best_values_dictionary) == len(best_values_legacy_dictionary) and len(best_values_legacy_dictionary) > 0) 

    failed_tests = 0
    passed_tests = 0
    for function_id in best_values_dictionary.keys():
        ftarget = best_values_dictionary[function_id]["ftarget"]
        ftarget_legacy = best_values_legacy_dictionary[function_id]["ftarget"]
        xopt = best_values_dictionary[function_id]["xopt"]
        xopt_legacy = best_values_legacy_dictionary[function_id]["xopt"]
        xopt_legacy = np.array([float("{:.6f}".format(x)) for x in xopt])
        try:
            assert(ftarget == float("{:.2f}".format(ftarget_legacy))), f"{function_id} failed a test: {ftarget=} is different from {ftarget_legacy=}"
            #assert(is_equal(xopt, xopt_legacy)), f"{function_id} failed a test: {xopt=} is different from {xopt_legacy=}"
            assert(is_equal(xopt, xopt_legacy)), f"{function_id} failed a test: xopt is different from xopt_legacy, seed {seeds[function_id]}, legacy_seed {seeds_legacy[function_id]}"
            passed_tests += 1 
            print('----------------------------------------------------')
            print(f"{function_id} has passed the test")
            print('----------------------------------------------------')
        except AssertionError as error:
            print('----------------------------------------------------')
            print(error)
            failed_tests += 1
            print(f"{ftarget=} is different from {ftarget_legacy=}")
            try:
                print(f"{xopt[:5]=}\n{xopt_legacy[:5]=}")
            except IndexError:
                print(f"{xopt=}\n{xopt_legacy=}")
            print('----------------------------------------------------')
    return failed_tests, passed_tests

if __name__ == "__main__":
    with open(best_values_legacy_path, "rb") as file_:
        best_values_legacy_dictionary = pickle.load(file_, encoding='latin1')
    with open(best_values_path, "r") as file_:
        best_values_string = file_.read()
    best_values_dictionary = parse_best_values_output_file(best_values_string)

    failed_tests, passed_tests = regression_test_best_values(best_values_dictionary, best_values_legacy_dictionary)
    if failed_tests > 0:
        print(f"Regression test best values has failed {failed_tests} tests, passed tests {passed_tests}")
    else:
        print(f"All tests passed, execution terminating with exit code {failed_tests}") 
    




