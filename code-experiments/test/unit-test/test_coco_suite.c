#include "coco.h"
#include "minunit.h"

/**
 * Tests the function coco_suite_get_problem.
 */
MU_TEST(test_coco_suite_get_problem) {

  coco_suite_t *suite = coco_suite("bbob-biobj", "instances: 1-10", "dimensions: 5");
  coco_problem_t *problem;

  problem = coco_suite_get_problem(suite, 0);
  mu_check(problem == NULL);

  problem = coco_suite_get_problem(suite, 1200);
  mu_check(problem != NULL);

  coco_problem_free(problem);
  coco_suite_free(suite);
}

/**
 * Tests the function coco_suite_encode_problem_index.
 */
MU_TEST(test_coco_suite_encode_problem_index) {

  coco_suite_t *suite;
  size_t index;

  suite = coco_suite("bbob", "year: 0000", NULL);
  index = coco_suite_encode_problem_index(suite, 13, 0, 10);
  mu_check(index == 205);
  coco_suite_free(suite);
}

/**
 * Run all tests in this file.
 */
MU_TEST_SUITE(test_all_coco_suite) {
  MU_RUN_TEST(test_coco_suite_encode_problem_index);
  MU_RUN_TEST(test_coco_suite_get_problem);
}

