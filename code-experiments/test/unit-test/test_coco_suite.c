/*
 * test_coco_suites.c
 *
 *  Created on: 22 nov. 2015
 *      Author: dejan
 */

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.h"

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
static void test_coco_suite_encode_problem_index(void **state) {

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

static int test_all_coco_suite(void) {

  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_coco_suite_encode_problem_index)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
