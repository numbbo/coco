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
 * Tests the function coco_suite_get_next_problem_index.
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



static char *convert_to_string(size_t *array) {

  size_t i = 0;
  char tmp[10];
  char *result;

  if (array == NULL)
    return NULL;

  result = coco_allocate_memory(1000 * sizeof(char));
  result[0] = '\0';

  while (array[i] > 0) {
    sprintf(tmp, "%lu,", array[i++]);
    strcat(result, tmp);
  }
  strcat(result, "0");

  return result;
}

static char *convert_to_string_with_newlines(char **array) {

  size_t i;
  char *result;

  if ((array == NULL) || (*array == NULL))
    return NULL;

  result = coco_allocate_memory(1000 * sizeof(char));
  result[0] = '\0';

  if (array) {
    for (i = 0; *(array + i); i++) {
      strcat(result, *(array + i));
      strcat(result, "\n");
    }
  }
  strcat(result, "\0");

  return result;
}

/**
 * Tests the function coco_suite_parse_ranges.
 */
static void test_coco_suite_parse_ranges(void **state) {

  size_t *result;
  char *converted_result;

  result = coco_suite_parse_ranges("", "", 1, 2);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_suite_parse_ranges(NULL, NULL, 1, 2);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_suite_parse_ranges("", "bla", 1, 2);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_suite_parse_ranges("bla", "", 1, 2);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_suite_parse_ranges("bla", "bla", 1, 2);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_suite_parse_ranges("1-3,5-6,7,-3,15-", "name", 1, 20);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "1,2,3,5,6,7,1,2,3,15,16,17,18,19,20,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_suite_parse_ranges("1-3,5-6", "name", 1, 2);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "1,2,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_suite_parse_ranges("1-3,5-6", "name", 0, 20);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "1,2,3,5,6,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_suite_parse_ranges("1-", "name", 1, 1200);
  assert_true(result[1000] == 0);
  assert_true(result[999] != 0);
  coco_free_memory(result);

  result = coco_suite_parse_ranges("1-2,1-", "name", 0, 0);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "1,2,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_suite_parse_ranges("1-4,5-8", "name", 2, 7);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "2,3,4,5,6,7,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_suite_parse_ranges("4-1", "name", 1, 20);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_suite_parse_ranges("-0", "name", 1, 20);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_suite_parse_ranges("-2,8-", "name", 3, 7);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_suite_parse_ranges("1-8", "name", 4, 8);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "4,5,6,7,8,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_suite_parse_ranges(",7", "name", 1, 20);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "7,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_suite_parse_ranges("7,", "name", 1, 20);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "7,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_suite_parse_ranges("-7-", "name", 1, 20);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_suite_parse_ranges("--7", "name", 1, 20);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_suite_parse_ranges("7--", "name", 1, 20);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_suite_parse_ranges("7-,-", "name", 5, 8);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "7,8,5,6,7,8,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_suite_parse_ranges("7-,,5", "name", 1, 8);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "7,8,5,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_suite_parse_ranges(",,", "name", 1, 20);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_suite_parse_ranges("1-8", "name", 5, 2);
  assert_true(result == NULL);
  coco_free_memory(result);

  (void)state; /* unused */
}


static int test_all_coco_suite(void) {

  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_coco_suite_encode_problem_index),
      cmocka_unit_test(test_coco_suite_parse_ranges)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
