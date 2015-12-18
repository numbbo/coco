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

static int test_all_coco_utilities(void) {

  const struct CMUnitTest tests[] =
  {
      cmocka_unit_test(test_coco_max_double_min_double),
      cmocka_unit_test(test_coco_round_double),
      cmocka_unit_test_setup_teardown(
          test_coco_create_remove_directory,
          setup_coco_create_remove_directory,
          teardown_coco_create_remove_directory)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
