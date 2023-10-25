#
# BBOB logger tests
#
# These tests ensure that the "old" and "new" logger code produce the same
# results. 
#

import pytest

import shutil
import logging
import numpy as np
from pathlib import Path

import cocoex
import cocopp


logger = logging.getLogger(__name__)
RESULTS = []


def run_experiment(suite, observer):
    logger.info(f"Running experiment: suite={suite.name} results={observer.result_folder}")

    budget_multiplier = 17 # Because Niko said so!

    np.random.seed(2342)
    problem_indices = np.arange(len(suite))
    for problem_index in problem_indices:
        problem = suite[problem_index]
        problem.observe_with(observer)

        x_start = np.random.uniform(-5.0, 5.0, problem.dimension)
        if problem.number_of_objectives == 1:
            problem._best_parameter("print")
            x_end = np.loadtxt("._bbob_problem_best_parameter.txt")
        else:
            x_end = np.random.uniform(-5.0, 5.0, problem.dimension)
         
        for α in np.linspace(0.0, 1.0, problem.dimension * budget_multiplier):
            x = α * x_end + (1.0 - α) * x_start
            problem(x)
    return observer.result_folder


def setup_module(module):
    logger.info("Creating test data.")
    exdata = Path("exdata/").absolute()
    if exdata.exists():
        logger.info("Removing stale 'exdata/' directory.")
        shutil.rmtree(exdata)

    suite_bbob = cocoex.Suite("bbob", "", "")
    obs_old = cocoex.Observer("bbob", f"result_folder: bbob_old")
    module.OLD = run_experiment(suite_bbob, obs_old)
    
    obs_new = cocoex.Observer("bbob", f"result_folder: bbob_new")
    module.NEW = run_experiment(suite_bbob, obs_new)


def teardown_module(module):
    if 1 < 0:
        logger.info("Removing test data.")
        shutil.rmtree("exdata/")


def test_cocopp_load():
    for result in [OLD, NEW]:
        logger.info(f"Loading '{result}' using cocopp.")
        res = cocopp.load(result)
        assert len(res) == 24 * 6 # 24 functions in 6 dimensions
        for ds in res:
            assert ds.suite_name == "bbob"
            assert ds.logger == "bbob"


def test_compare_results():
    logger.info("Comparing results.")
    old_res = cocopp.load(OLD)
    new_res = cocopp.load(NEW)

    assert len(old_res) == len(new_res)
    for old_d, new_d in zip(old_res, new_res):
        assert np.allclose(old_d.evals, new_d.evals, equal_nan=True)
        assert np.allclose(old_d.funvals, new_d.funvals, equal_nan=True)
        assert np.allclose(old_d.ert, new_d.ert, equal_nan=True)
