#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.h"

#if 0

/* This is how wrapping would look like if it worked */

static size_t __wrap_coco_suite_encode_problem_index(coco_suite_t *suite,
                                                     const size_t function_idx,
                                                     const size_t dimension_idx,
                                                     const size_t instance_idx) {

  printf("INFO: function __wrap_coco_suite_encode_problem_index.\n");

  check_expected(function_idx);
  check_expected(dimension_idx);
  check_expected(instance_idx);

  (void)suite; /* unused */
  return (size_t) mock();
}

/**
 * Tests the function coco_suite_get_next_problem_index.
 */
static void test_coco_suite_encode_problem_index_with_wrapping(void **state) {

  coco_suite_t *suite;
  size_t index;
  size_t function_idx = 13, dimension_idx = 0, instance_idx = 10;

  suite = coco_suite("bbob", NULL, NULL);

  expect_value(__wrap_coco_suite_encode_problem_index, function_idx, 13);
  expect_value(__wrap_coco_suite_encode_problem_index, dimension_idx, 0);
  expect_value(__wrap_coco_suite_encode_problem_index, instance_idx, 10);
  will_return(__wrap_coco_suite_encode_problem_index, 205);

  index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  assert_true(index == 205);

  coco_suite_free(suite);

  (void)state; /* unused */
}
#endif

/**
 * Tests the function coco_suite_get_problem.
 */
static void test_coco_suite_get_problem(void **state) {

  coco_suite_t *suite = coco_suite("bbob-biobj", NULL, "dimensions: 5");
  coco_problem_t *problem;

  problem = coco_suite_get_problem(suite, 0);
  assert_true(problem == NULL);

  problem = coco_suite_get_problem(suite, 1200);
  assert_true(problem != NULL);

  coco_problem_free(problem);
  coco_suite_free(suite);

  (void)state; /* unused */
}

/**
 * Tests the function coco_suite_encode_problem_index.
 */
static void test_coco_suite_encode_problem_index(void **state) {

  coco_suite_t *suite;
  size_t index;

  suite = coco_suite("bbob", NULL, NULL);
  index = coco_suite_encode_problem_index(suite, 13, 0, 10);
  assert_true(index == 205);
  coco_suite_free(suite);

  (void)state; /* unused */
}

static int test_all_coco_suite(void) {

  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_coco_suite_encode_problem_index),
      cmocka_unit_test(test_coco_suite_get_problem)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
