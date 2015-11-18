/*
 * test_coco_utilities.c
 *
 *  Created on: 18 nov. 2015
 *      Author: dejan
 */

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.c"

static void test_coco_max_double_min_double(void **state)
{
	double first_value = 5.0;
	double second_value = 6.0;

	double max_value, min_value;

  max_value = coco_max_double(first_value, second_value);
	assert_true(max_value == second_value);

  min_value = coco_min_double(first_value, second_value);
  assert_true(min_value == first_value);

	(void)state; /* unused */
}

static void test_coco_round_double(void **state)
{

  double input_value = 5.4;

  double round_value = coco_round_double(input_value);
  assert_true(round_value == 5);

  input_value = 5.5;
  round_value = coco_round_double(input_value);
  assert_true(round_value == 6);

  (void)state; /* unused */
}

static int test_all_coco_utilities(void)
{
  const struct CMUnitTest tests[] =
  {
      cmocka_unit_test(test_coco_max_double_min_double),
      cmocka_unit_test(test_coco_round_double),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
