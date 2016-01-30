#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.h"
#include <time.h>

/**
 * Tests the function coco_set_log_level.
 */
static void test_coco_set_log_level(void **state) {

  char *previous_log_level;

  /* Check whether the default set to COCO_INFO */
  assert(strcmp(coco_set_log_level(""), "info") == 0);

  /* Check whether the method works */
  previous_log_level = coco_strdup(coco_set_log_level("error"));
  assert(strcmp(previous_log_level, "info") == 0);
  assert(strcmp(coco_set_log_level(""), "error") == 0);
  coco_free_memory(previous_log_level);

  previous_log_level = coco_strdup(coco_set_log_level("warning"));
  assert(strcmp(previous_log_level, "error") == 0);
  assert(strcmp(coco_set_log_level(""), "warning") == 0);
  coco_free_memory(previous_log_level);

  previous_log_level = coco_strdup(coco_set_log_level("debug"));
  assert(strcmp(previous_log_level, "warning") == 0);
  assert(strcmp(coco_set_log_level(""), "debug") == 0);
  coco_free_memory(previous_log_level);

  previous_log_level = coco_strdup(coco_set_log_level("info"));
  assert(strcmp(previous_log_level, "debug") == 0);
  assert(strcmp(coco_set_log_level(""), "info") == 0);
  coco_free_memory(previous_log_level);

  /* An invalid argument shouldn't change the current value */
  previous_log_level = coco_strdup(coco_set_log_level("bla"));
  assert(strcmp(previous_log_level, "info") == 0);
  assert(strcmp(coco_set_log_level(""), "info") == 0);
  coco_free_memory(previous_log_level);

  (void)state; /* unused */
}

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
  exists = coco_directory_exists(path_string);
  assert_false(exists);

  coco_create_directory(path_string);
  exists = coco_directory_exists(path_string);
  assert_true(exists);

  /* Calling it again to check the handling if the path does exist. */
  coco_create_directory(path_string);

  coco_remove_directory(path_string);
  exists = coco_directory_exists(path_string);
  assert_false(exists);

  /* Calling it again to check the handling if the path does not exist. */
  coco_remove_directory(path_string);
}

/**
 * Tests the functions coco_double_max and coco_double_min.
 */
static void test_coco_double_max_min(void **state) {

  double first_value = 5.0;
	double second_value = 6.0;

	double max_value, min_value;

  max_value = coco_double_max(first_value, second_value);
	assert_true(max_value == second_value);

  min_value = coco_double_min(first_value, second_value);
  assert_true(min_value == first_value);

	(void)state; /* unused */
}

/**
 * Tests the function coco_double_round.
 */
static void test_coco_double_round(void **state) {

  double input_value = 5.4;

  double round_value = coco_double_round(input_value);
  assert_true(round_value == 5);

  input_value = 5.5;
  round_value = coco_double_round(input_value);
  assert_true(round_value == 6);

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
      cmocka_unit_test(test_coco_set_log_level),
      cmocka_unit_test(test_coco_double_max_min),
      cmocka_unit_test(test_coco_double_round),
      cmocka_unit_test(test_coco_string_split),
      cmocka_unit_test_setup_teardown(
          test_coco_create_remove_directory,
          setup_coco_create_remove_directory,
          teardown_coco_create_remove_directory)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
