#include "../../src/coco.h"
#include "minunit_c89.h"

/**
 * Tests the creation of new instances of the bi-objective suites.
 */
MU_TEST(test_biobj_utilities_instances) {

  coco_suite_t *suite = coco_suite("bbob-noisy", "instances: 16-20", "");
  coco_problem_t *problem;

  problem = coco_suite_get_next_problem(suite, NULL);
  mu_check(coco_problem_get_suite_dep_instance(problem) == 16);

  problem = coco_suite_get_next_problem(suite, NULL);
  mu_check(coco_problem_get_suite_dep_instance(problem) == 17);

  problem = coco_suite_get_next_problem(suite, NULL);
  mu_check(coco_problem_get_suite_dep_instance(problem) == 18);

  problem = coco_suite_get_next_problem(suite, NULL);
  mu_check(coco_problem_get_suite_dep_instance(problem) == 19);

  problem = coco_suite_get_next_problem(suite, NULL);
  mu_check(coco_problem_get_suite_dep_instance(problem) == 20);

  coco_suite_free(suite);
}

/**
 * Run all tests in this file.
 */
MU_TEST_SUITE(test_all_biobj_utilities) {
  MU_RUN_TEST(test_biobj_utilities_instances);
}

