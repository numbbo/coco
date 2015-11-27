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

static long __wrap_suite_bbob2009_get_next_problem_index(
    long problem_index, const char *selection_descriptor) {

  printf("INFO: function __wrap_suite_bbob2009_get_next_problem_index.\n");

  check_expected(problem_index);

  return (long) mock();
}

/**
 * Tests the function coco_suite_get_next_problem_index.
 */
static void test_coco_suite_get_next_problem_index(void **state) {

  long prob_index;

  expect_string(__wrap_suite_bbob2009_get_next_problem_index, problem_index, -1);
  will_return(__wrap_suite_bbob2009_get_next_problem_index, 2);

  prob_index = coco_suite_get_next_problem_index("suite_bbob2009", -1, NULL);

  assert_true(prob_index == 2);

  (void)state; /* unused */
}

static int test_all_coco_suites(void) {

  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_coco_suite_get_next_problem_index)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
