#include "minunit.h"

#include "coco.c"

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
 * Tests the function coco_set_log_level.
 */
MU_TEST(test_coco_set_log_level) {

  char *previous_log_level, *tmp_log_level;

  /* Remember the log level before all these tests */
  previous_log_level = coco_strdup(coco_set_log_level("info"));

  /* Check whether the level is set to COCO_INFO */
  mu_check(strcmp(coco_set_log_level(""), "info") == 0);

  /* Check whether the method works */
  tmp_log_level = coco_strdup(coco_set_log_level("error"));
  mu_check(strcmp(tmp_log_level, "info") == 0);
  mu_check(strcmp(coco_set_log_level(""), "error") == 0);
  coco_free_memory(tmp_log_level);

  tmp_log_level = coco_strdup(coco_set_log_level("warning"));
  mu_check(strcmp(tmp_log_level, "error") == 0);
  mu_check(strcmp(coco_set_log_level(""), "warning") == 0);
  coco_free_memory(tmp_log_level);

  tmp_log_level = coco_strdup(coco_set_log_level("debug"));
  mu_check(strcmp(tmp_log_level, "warning") == 0);
  mu_check(strcmp(coco_set_log_level(""), "debug") == 0);
  coco_free_memory(tmp_log_level);

  tmp_log_level = coco_strdup(coco_set_log_level("info"));
  mu_check(strcmp(tmp_log_level, "debug") == 0);
  mu_check(strcmp(coco_set_log_level(""), "info") == 0);
  coco_free_memory(tmp_log_level);

  /* An invalid argument shouldn't change the current value */
  coco_set_log_level("error");
  tmp_log_level = coco_strdup(coco_set_log_level("bla"));
  mu_check(strcmp(tmp_log_level, "error") == 0);
  mu_check(strcmp(coco_set_log_level(""), "error") == 0);
  coco_free_memory(tmp_log_level);

  /* Restore the log level */
  coco_set_log_level(previous_log_level);
  coco_free_memory(previous_log_level);
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
  new_string = coco_strconcat(*string, date_time_string);
  coco_free_memory(date_time_string);
  coco_free_memory(*string);
  *string = new_string;
}

/*
 * Tests creating and removing directory.
 */
MU_TEST(test_coco_create_remove_directory) {

  int exists;
  char *path_string = NULL;

  path_string = coco_strdup("temp");
  create_time_string(&path_string);

  mu_check(path_string != NULL);

  /* At the beginning the path should not exist. */
  exists = coco_directory_exists(path_string);
  mu_check(!exists);

  coco_create_directory(path_string);
  exists = coco_directory_exists(path_string);
  mu_check(exists);

  /* Calling it again to check the handling if the path does exist. */
  coco_create_directory(path_string);

  coco_remove_directory(path_string);
  exists = coco_directory_exists(path_string);
  mu_check(!exists);

  /* Calling it again to check the handling if the path does not exist. */
  coco_remove_directory(path_string);

  coco_free_memory(path_string);
}

/**
 * Tests the functions coco_double_max and coco_double_min.
 */
MU_TEST(test_coco_double_max_min) {

  double first_value = 5.0;
  double second_value = 6.0;

  double max_value, min_value;

  max_value = coco_double_max(first_value, second_value);
  mu_check(max_value == second_value);

  min_value = coco_double_min(first_value, second_value);
  mu_check(min_value == first_value);
}

/**
 * Tests the function coco_double_round.
 */
MU_TEST(test_coco_double_round) {

  double input_value = 5.4;

  double round_value = coco_double_round(input_value);
  mu_check(round_value == 5);

  input_value = 5.5;
  round_value = coco_double_round(input_value);
  mu_check(round_value == 6);

  input_value = 5;
  round_value = coco_double_round(input_value);
  mu_check(round_value == 5);

  input_value = 5 - 0.0000000000000001;
  round_value = coco_double_round(input_value);
  mu_check(round_value == 5);

  input_value = 5 + 0.0000000000000001;
  round_value = coco_double_round(input_value);
  mu_check(round_value == 5);
}

/**
 * Tests the function coco_double_round_up_with_precision.
 */
MU_TEST(test_coco_double_round_up_with_precision) {

  double input_value = 5.0024;
  double round_value = coco_double_round_up_with_precision(input_value, 0.00001);
  mu_check(coco_double_almost_equal(round_value, 5.00240, 1e-10));

  round_value = coco_double_round_up_with_precision(input_value, 0.0001);
  mu_check(coco_double_almost_equal(round_value, 5.0024, 1e-10));

  round_value = coco_double_round_up_with_precision(input_value, 0.001);
  mu_check(coco_double_almost_equal(round_value, 5.003, 1e-10));

  round_value = coco_double_round_up_with_precision(input_value, 0.01);
  mu_check(coco_double_almost_equal(round_value, 5.01, 1e-10));

  input_value = -5.0024;
  round_value = coco_double_round_up_with_precision(input_value, 0.00001);
  mu_check(coco_double_almost_equal(round_value, -5.00240, 1e-10));

  round_value = coco_double_round_up_with_precision(input_value, 0.0001);
  mu_check(coco_double_almost_equal(round_value, -5.0024, 1e-10));

  round_value = coco_double_round_up_with_precision(input_value, 0.001);
  mu_check(coco_double_almost_equal(round_value, -5.002, 1e-10));

  round_value = coco_double_round_up_with_precision(input_value, 0.01);
  mu_check(coco_double_almost_equal(round_value, -5, 1e-10));

  input_value = 0;
  round_value = coco_double_round_up_with_precision(input_value, 0.001);
  mu_check(coco_double_almost_equal(round_value, 0, 1e-10));

  input_value = -5.9999;
  round_value = coco_double_round_up_with_precision(input_value, 0.00001);
  mu_check(coco_double_almost_equal(round_value, -5.9999, 1e-10));

  round_value = coco_double_round_up_with_precision(input_value, 0.0001);
  mu_check(coco_double_almost_equal(round_value, -5.9999, 1e-10));

  round_value = coco_double_round_up_with_precision(input_value, 0.001);
  mu_check(coco_double_almost_equal(round_value, -5.999, 1e-10));

  input_value = 5.0001;
  round_value = coco_double_round_up_with_precision(input_value, 0.00001);
  mu_check(coco_double_almost_equal(round_value, 5.0001, 1e-10));

  round_value = coco_double_round_up_with_precision(input_value, 0.0001);
  mu_check(coco_double_almost_equal(round_value, 5.0001, 1e-10));

  round_value = coco_double_round_up_with_precision(input_value, 0.001);
  mu_check(coco_double_almost_equal(round_value, 5.001, 1e-10));

  input_value = 1e-8;
  round_value = coco_double_round_up_with_precision(input_value, 0.00001);
  mu_check(coco_double_almost_equal(round_value, 0.00001, 1e-10));

  round_value = coco_double_round_up_with_precision(input_value, 0.0001);
  mu_check(coco_double_almost_equal(round_value, 0.0001, 1e-10));

  round_value = coco_double_round_up_with_precision(input_value, 0.001);
  mu_check(coco_double_almost_equal(round_value, 0.001, 1e-10));

  input_value = 123;
  round_value = coco_double_round_up_with_precision(input_value, 20);
  mu_check(coco_double_almost_equal(round_value, 140, 1e-10));

  round_value = coco_double_round_up_with_precision(input_value, 120);
  mu_check(coco_double_almost_equal(round_value, 240, 1e-10));

  round_value = coco_double_round_up_with_precision(input_value, 123);
  mu_check(coco_double_almost_equal(round_value, 123, 1e-10));
}

/**
 * Tests the function coco_string_split.
 */
MU_TEST(test_coco_string_split) {

  char **result;
  char *converted_result;
  size_t i;

  result = coco_string_split("1-3,5-6,7,-3,15-", ',');
  converted_result = convert_to_string_with_newlines(result);
  mu_check(converted_result);
  mu_check(strcmp(converted_result, "1-3\n5-6\n7\n-3\n15-\n") == 0);
  for (i = 0; *(result + i); i++) {
    coco_free_memory(*(result + i));
  }
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_split(",,,a,,b,c,d,e,,,,,f,", ',');
  converted_result = convert_to_string_with_newlines(result);
  mu_check(converted_result);
  mu_check(strcmp(converted_result, "a\nb\nc\nd\ne\nf\n") == 0);
  for (i = 0; *(result + i); i++) {
    coco_free_memory(*(result + i));
  }
  coco_free_memory(result);
  coco_free_memory(converted_result);
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
    sprintf(tmp, "%lu,", (unsigned long) array[i++]);
    strcat(result, tmp);
  }
  strcat(result, "0");

  return result;
}

/**
 * Tests the function coco_string_parse_ranges.
 */
MU_TEST(test_coco_string_parse_ranges) {

  size_t *result;
  char *converted_result;

  result = coco_string_parse_ranges("", 1, 2, "", 100);
  mu_check(result == NULL);
  coco_free_memory(result);

  result = coco_string_parse_ranges(NULL, 1, 2, NULL, 100);
  mu_check(result == NULL);
  coco_free_memory(result);

  result = coco_string_parse_ranges("", 1, 2, "bla", 100);
  mu_check(result == NULL);
  coco_free_memory(result);

  result = coco_string_parse_ranges("bla", 1, 2, "", 100);
  mu_check(result == NULL);
  coco_free_memory(result);

  result = coco_string_parse_ranges("bla", 1, 2, "bla", 100);
  mu_check(result == NULL);
  coco_free_memory(result);

  result = coco_string_parse_ranges("1-3,5-6,7,-3,15-", 1, 20, "name", 100);
  converted_result = convert_to_string(result);
  mu_check(converted_result);
  mu_check(strcmp(converted_result, "1,2,3,5,6,7,1,2,3,15,16,17,18,19,20,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_parse_ranges("1-3,5-6", 1, 2, "name", 100);
  converted_result = convert_to_string(result);
  mu_check(converted_result);
  mu_check(strcmp(converted_result, "1,2,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_parse_ranges("1-3,5-6", 0, 20, "name", 100);
  converted_result = convert_to_string(result);
  mu_check(converted_result);
  mu_check(strcmp(converted_result, "1,2,3,5,6,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_parse_ranges("1-", 1, 1200, "name", 1000);
  mu_check(result[1000] == 0);
  mu_check(result[999] != 0);
  coco_free_memory(result);

  result = coco_string_parse_ranges("1-2,1-", 0, 0, "name", 1000);
  converted_result = convert_to_string(result);
  mu_check(converted_result);
  mu_check(strcmp(converted_result, "1,2,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_parse_ranges("1-4,5-8", 2, 7, "name", 1000);
  converted_result = convert_to_string(result);
  mu_check(converted_result);
  mu_check(strcmp(converted_result, "2,3,4,5,6,7,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_parse_ranges("4-1", 1, 20, "name", 1000);
  mu_check(result == NULL);
  coco_free_memory(result);

  result = coco_string_parse_ranges("-0", 1, 20, "name", 1000);
  mu_check(result == NULL);
  coco_free_memory(result);

  result = coco_string_parse_ranges("-2,8-", 3, 7, "name", 1000);
  mu_check(result == NULL);
  coco_free_memory(result);

  result = coco_string_parse_ranges("1-8", 4, 8, "name", 1000);
  converted_result = convert_to_string(result);
  mu_check(converted_result);
  mu_check(strcmp(converted_result, "4,5,6,7,8,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_parse_ranges(",7", 1, 20, "name", 1000);
  converted_result = convert_to_string(result);
  mu_check(converted_result);
  mu_check(strcmp(converted_result, "7,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_parse_ranges("7,", 1, 20, "name", 1000);
  converted_result = convert_to_string(result);
  mu_check(converted_result);
  mu_check(strcmp(converted_result, "7,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_parse_ranges("-7-", 1, 20, "name", 1000);
  mu_check(result == NULL);
  coco_free_memory(result);

  result = coco_string_parse_ranges("--7", 1, 20, "name", 1000);
  mu_check(result == NULL);
  coco_free_memory(result);

  result = coco_string_parse_ranges("7--", 1, 20, "name", 1000);
  mu_check(result == NULL);
  coco_free_memory(result);

  result = coco_string_parse_ranges("7-,-", 5, 8, "name", 1000);
  converted_result = convert_to_string(result);
  mu_check(converted_result);
  mu_check(strcmp(converted_result, "7,8,5,6,7,8,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_parse_ranges("7-,,5", 1, 8, "name", 1000);
  converted_result = convert_to_string(result);
  mu_check(converted_result);
  mu_check(strcmp(converted_result, "7,8,5,0") == 0);
  coco_free_memory(result);
  coco_free_memory(converted_result);

  result = coco_string_parse_ranges(",,", 1, 20, "name", 1000);
  mu_check(result == NULL);
  coco_free_memory(result);

  result = coco_string_parse_ranges("1-8", 5, 2, "name", 1000);
  mu_check(result == NULL);
  coco_free_memory(result);
}

/**
 * Tests the function coco_option_keys.
 */
MU_TEST(test_coco_option_keys) {

  coco_option_keys_t *option_keys = NULL;

  option_keys = coco_option_keys("key");
  mu_check(option_keys->count == 1);
  coco_option_keys_free(option_keys);

  option_keys = coco_option_keys("key: ");
  mu_check(option_keys->count == 1);
  coco_option_keys_free(option_keys);

  option_keys = coco_option_keys("key1 key2: ");
  mu_check(option_keys->count == 1);
  coco_option_keys_free(option_keys);

  option_keys = coco_option_keys("key1: value1 key2");
  /* In this case we would rather have detected two keys, but this should also trigger a warning,
   * which is OK. */
  mu_check(option_keys->count == 1);
  coco_option_keys_free(option_keys);

  option_keys = coco_option_keys("key1: value1 key2: value2");
  mu_check(option_keys->count == 2);
  coco_option_keys_free(option_keys);

  option_keys = coco_option_keys("key: \"A multi-word value\"");
  mu_check(option_keys->count == 1);
  coco_option_keys_free(option_keys);

  option_keys = coco_option_keys("key: \"A multi-word value with a colon (:)\"");
  mu_check(option_keys->count == 1);
  coco_option_keys_free(option_keys);

  option_keys = coco_option_keys("\"key: value\"");
  mu_check(option_keys->count == 1);
  coco_option_keys_free(option_keys);
}

/**
 * Tests the function coco_is_nan.
 */
MU_TEST(test_coco_is_nan) {

  mu_check(coco_is_nan(NAN));
  mu_check(!coco_is_nan(1e100));
}

/**
 * Tests the function coco_is_inf.
 */
MU_TEST(test_coco_is_inf) {

  mu_check(coco_is_inf(INFINITY));
  mu_check(coco_is_inf(-INFINITY));
  mu_check(coco_is_inf(2*INFINITY));
  mu_check(coco_is_inf(-2*INFINITY));
  mu_check(!coco_is_inf(NAN));
}

/**
 * Tests the function coco_is_orthogonal.
 */
MU_TEST(test_coco_is_orthogonal) {

  double *M;

  M = coco_allocate_vector(4);

  /* Case of an orthogonal diagonal matrix (identity matrix) */
  M[0] = 1.;
  M[1] = 0.;
  M[2] = 0.;
  M[3] = 1.;
  mu_check(coco_is_orthogonal(M, 2, 2));

  /* Case of a non-orthogonal square matrix */
  M[0] = 1.;
  M[1] = 0.01;
  M[2] = 0.;
  M[3] = 1.;
  mu_check(!coco_is_orthogonal(M, 2, 2));

  /* Case of an orthogonal diagonal matrix (almost identity matrix)
   * such that the function "coco_double_almost_equal()" is tested
   */
  M[0] = 1. + 1e-10;
  M[1] = 0.;
  M[2] = 0.;
  M[3] = 1. + 1e-10;
  mu_check(coco_is_orthogonal(M, 2, 2));

  coco_free_memory(M);
}

/**
 * Run all tests in this file.
 */
int main(void) {
  MU_RUN_TEST(test_coco_is_nan);
  MU_RUN_TEST(test_coco_is_inf);
  MU_RUN_TEST(test_coco_set_log_level);
  MU_RUN_TEST(test_coco_double_max_min);
  MU_RUN_TEST(test_coco_double_round);
  MU_RUN_TEST(test_coco_double_round_up_with_precision);
  MU_RUN_TEST(test_coco_string_split);
  MU_RUN_TEST(test_coco_option_keys);
  MU_RUN_TEST(test_coco_string_parse_ranges);
  MU_RUN_TEST(test_coco_create_remove_directory);
  MU_RUN_TEST(test_coco_is_orthogonal);

  MU_REPORT();

  return minunit_status;
}
