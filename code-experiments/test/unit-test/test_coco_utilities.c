#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.h"
#include <time.h>

static void create_time_string(char **string) {

  time_t date_time;
  size_t len;
  const size_t date_time_len = 24;
  char date_time_format[] = "_%Y_%m_%d_%H_%M_%S";
  char *date_time_string;
  struct tm* date_time_info;
  char *new_string;

  /* Retrieve the current time as string of the defined format */
  time(&date_time);
  date_time_info = localtime(&date_time);
  date_time_string = coco_allocate_memory(sizeof(char) * date_time_len);
  date_time_string[0] = '\1';

  len = strftime(date_time_string, date_time_len, date_time_format, date_time_info);
  if (len == 0 && date_time_string[0] != '\0') {
    coco_error("create_time_string(): cannot decode local time string");
    return; /* Never reached */
  }

  /* Produce the new string */
  new_string = coco_allocate_memory(sizeof(char) * (strlen(*string) + date_time_len));
  new_string = coco_strconcat(*string, date_time_string);
  coco_free_memory(date_time_string);
  coco_free_memory(*string);
  *string = new_string;
}

/*
 * Setup for creating directory test.
 */
static int setup_coco_create_remove_directory(void **state) {

  char *path_string;

  path_string = coco_strdup("temp");
  create_time_string(&path_string);

  assert_non_null(path_string);

  *state = (void*)path_string;

  return 0;
}

/*
 * Tear down for creating directory test.
 */
static int teardown_coco_create_remove_directory(void **state) {

  coco_free_memory(*state);
  return 0;
}

/*
 * Tests creating and removing directory.
 */
static void test_coco_create_remove_directory(void **state) {

  int exists;
  char *path_string = (char *)*state;

  /* At the beginning the path should not exist. */
  exists = coco_path_exists(path_string);
  assert_false(exists);

  coco_create_directory(path_string);
  exists = coco_path_exists(path_string);
  assert_true(exists);

  /* Calling it again to check the handling if the path does exist. */
  coco_create_directory(path_string);

  coco_remove_directory(path_string);
  exists = coco_path_exists(path_string);
  assert_false(exists);

  /* Calling it again to check the handling if the path does not exist. */
  coco_remove_directory(path_string);
}

/**
 * Tests the functions coco_max_double and coco_min_double.
 */
static void test_coco_max_double_min_double(void **state) {

  double first_value = 5.0;
	double second_value = 6.0;

	double max_value, min_value;

  max_value = coco_max_double(first_value, second_value);
	assert_true(max_value == second_value);

  min_value = coco_min_double(first_value, second_value);
  assert_true(min_value == first_value);

	(void)state; /* unused */
}

/**
 * Tests the function coco_round_double.
 */
static void test_coco_round_double(void **state) {

  double input_value = 5.4;

  double round_value = coco_round_double(input_value);
  assert_true(round_value == 5);

  input_value = 5.5;
  round_value = coco_round_double(input_value);
  assert_true(round_value == 6);

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
 * Tests the function coco_string_get_numbers_from_ranges.
 */
static void test_coco_string_get_numbers_from_ranges(void **state) {

  size_t *result;
  char *converted_result;

  result = coco_string_get_numbers_from_ranges("", "", 1, 2);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_string_get_numbers_from_ranges(NULL, NULL, 1, 2);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_string_get_numbers_from_ranges("", "bla", 1, 2);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_string_get_numbers_from_ranges("bla", "", 1, 2);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_string_get_numbers_from_ranges("bla", "bla", 1, 2);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_string_get_numbers_from_ranges("1-3,5-6,7,-3,15-", "name", 1, 20);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "1,2,3,5,6,7,1,2,3,15,16,17,18,19,20,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_get_numbers_from_ranges("1-3,5-6", "name", 1, 2);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "1,2,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_get_numbers_from_ranges("1-3,5-6", "name", 0, 20);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "1,2,3,5,6,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_get_numbers_from_ranges("1-", "name", 1, 1200);
  assert_true(result[1000] == 0);
  assert_true(result[999] != 0);
  coco_free_memory(result);

  result = coco_string_get_numbers_from_ranges("1-2,1-", "name", 0, 0);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "1,2,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_get_numbers_from_ranges("1-4,5-8", "name", 2, 7);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "2,3,4,5,6,7,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_get_numbers_from_ranges("4-1", "name", 1, 20);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_string_get_numbers_from_ranges("-0", "name", 1, 20);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_string_get_numbers_from_ranges("-2,8-", "name", 3, 7);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_string_get_numbers_from_ranges("1-8", "name", 4, 8);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "4,5,6,7,8,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_get_numbers_from_ranges(",7", "name", 1, 20);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "7,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_get_numbers_from_ranges("7,", "name", 1, 20);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "7,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_get_numbers_from_ranges("-7-", "name", 1, 20);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_string_get_numbers_from_ranges("--7", "name", 1, 20);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_string_get_numbers_from_ranges("7--", "name", 1, 20);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_string_get_numbers_from_ranges("7-,-", "name", 5, 8);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "7,8,5,6,7,8,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_get_numbers_from_ranges("7-,,5", "name", 1, 8);
  converted_result = convert_to_string(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "7,8,5,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_get_numbers_from_ranges(",,", "name", 1, 20);
  assert_true(result == NULL);
  coco_free_memory(result);

  result = coco_string_get_numbers_from_ranges("1-8", "name", 5, 2);
  assert_true(result == NULL);
  coco_free_memory(result);

  (void)state; /* unused */
}

/**
 * Tests the function coco_string_split.
 */
static void test_coco_string_split(void **state) {

  char **result;
  char *converted_result;

  result = coco_string_split("1-3,5-6,7,-3,15-", ',');
  converted_result = convert_to_string_with_newlines(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "1-3\n5-6\n7\n-3\n15-\n") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_split(",,,a,,b,c,d,e,,,,,f,", ',');
  converted_result = convert_to_string_with_newlines(result);
  assert_true(converted_result);
  assert_true(strcmp(converted_result, "a\nb\nc\nd\ne\nf\n") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  (void)state; /* unused */
}


static int test_all_coco_utilities(void) {

  const struct CMUnitTest tests[] =
  {
      cmocka_unit_test(test_coco_max_double_min_double),
      cmocka_unit_test(test_coco_round_double),
      cmocka_unit_test(test_coco_string_split),
      cmocka_unit_test(test_coco_string_get_numbers_from_ranges),
      cmocka_unit_test_setup_teardown(
          test_coco_create_remove_directory,
          setup_coco_create_remove_directory,
          teardown_coco_create_remove_directory)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
