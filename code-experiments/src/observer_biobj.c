#include "coco.h"
#include "coco_internal.h"

#include "coco_utilities.c"
#include "mo_generics.c"

/* List of implemented indicators */
#define OBSERVER_BIOBJ_NUMBER_OF_INDICATORS 1
const char *OBSERVER_BIOBJ_INDICATORS[OBSERVER_BIOBJ_NUMBER_OF_INDICATORS] = { "hyp" };
const size_t OBSERVER_BIOBJ_MAX_STR_LENGTH = 100;

/* Logging nondominated solutions mode */
typedef enum {
  NONE, FINAL, ALL
} observer_biobj_log_nondom_e;

/* Data for the biobjective observer */
typedef struct {
  observer_biobj_log_nondom_e log_mode;

  int include_decision_variables;
  int compute_indicators;
  int produce_all_data;

  char ***best_values_matrix[OBSERVER_BIOBJ_NUMBER_OF_INDICATORS];
  size_t best_values_count;

  /* Information on the previous logged problem */
  int previous_function;

} observer_biobj_t;

static coco_problem_t *logger_biobj(coco_observer_t *self, coco_problem_t *problem);

/**
 * Allocates memory for a matrix with number_of_rows rows and number_of_columns columns of strings of
 * maximal size max_length_of_string.
 */
static char ***observer_biobj_allocate_matrix_of_strings(const size_t number_of_rows,
                                                         const size_t number_of_columns,
                                                         const size_t max_length_of_string) {

  char ***matrix_of_strings;
  size_t i, j;

  matrix_of_strings = malloc(number_of_rows * sizeof(char**));
  if (matrix_of_strings == NULL)
    coco_error("observer_biobj_allocate_matrix_of_strings() failed");

  for (i = 0; i < number_of_rows; i++) {
    matrix_of_strings[i] = malloc(number_of_columns * sizeof(char*));
    for (j = 0; j < number_of_columns; j++) {
      matrix_of_strings[i][j] = malloc(max_length_of_string * sizeof(char));
    }
  }

  return matrix_of_strings;
}

/**
 * Frees the memory occupied by the matrix_of_strings matrix of string with number_of_rows rows and
 * cnumber_of_columns columns.
 */
static void observer_biobj_free_matrix_of_strings(char ***matrix_of_strings,
                                                  const size_t number_of_rows,
                                                  const size_t number_of_columns) {

  size_t i, j;

  for (i = 0; i < number_of_rows; i++) {
    for (j = 0; j < number_of_columns; j++) {
      free(matrix_of_strings[i][j]);
    }
    free(matrix_of_strings[i]);
  }
  free(matrix_of_strings);
}

/**
 * Counts and returns the number of lines in the already opened file.
 */
static size_t observer_biobj_get_number_of_lines_in_file(FILE *file) {

  int ch;
  size_t number_of_lines = 0;

  /* Count the number of lines */
  do {
    ch = fgetc(file);
    if (ch == '\n')
      number_of_lines++;
  } while (ch != EOF);
  /* Add 1 if the last line doesn't end with \n */
  if (ch != '\n' && number_of_lines != 0)
    number_of_lines++;

  /* Return to the beginning of the file */
  rewind(file);

  return number_of_lines;
}

/**
 * Assumes the input file contains number_of_lines pairs of strings separated by tabs and that each line is
 * of maximal length max_length_of_string. Returns the strings in the form of a number_of_lines x 2 matrix.
 */
static char ***observer_biobj_get_string_pairs_from_file(FILE *file,
                                                         const size_t number_of_lines,
                                                         const size_t max_length_of_string) {

  size_t i;
  char ***matrix_of_strings;

  /* Prepare the matrix */
  matrix_of_strings = observer_biobj_allocate_matrix_of_strings(number_of_lines, 2, max_length_of_string);
  for (i = 0; i < number_of_lines; i++) {
    fscanf(file, "%s\t%[^\n]", matrix_of_strings[i][0], matrix_of_strings[i][1]);
  }

  return matrix_of_strings;
}

/**
 * Converts string (char *) to double. Does not check for underflow or overflow, ignores any trailing
 * characters.
 */
static double observer_biobj_string_to_double(const char *string) {
  double result;
  char *err;

  result = strtod(string, &err);
  if (result == 0 && string == err) {
    coco_error("observer_biobj_string_to_double() failed");
  }

  return result;
}

/**
 * Scans the input matrix to find the first value that matches the given key (looks for the key in
 * key_column column and for the value in value_column column). If the key is not found, it returns
 * default_value.
 */
static double observer_biobj_get_matching_double_value(char ***matrix_of_strings,
                                                       const char *key,
                                                       const size_t number_of_rows,
                                                       const size_t key_column,
                                                       const size_t value_column,
                                                       const double default_value) {

  size_t i;

  for (i = 0; i < number_of_rows; i++) {
    /* The given key matches the key in the matrix */
    if (strcmp(matrix_of_strings[i][key_column], key) == 0) {
      /* Return the value in the same row */
      return observer_biobj_string_to_double(matrix_of_strings[i][value_column]);
    }
  }

  /* The key was not found, therefore the default value is returned */
  return default_value;
}

/* Returns the best known value for indicator_name matching the given key if the key is found, and raises an
 * error otherwise.  */
static double observer_biobj_read_best_value(const observer_biobj_t *self,
                                             const char *indicator_name,
                                             const char *key) {

  size_t i;
  double best_value;
  double error_value = -1;
  double error_accuracy = 1e-8;

  for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
    if (strcmp(OBSERVER_BIOBJ_INDICATORS[i], indicator_name) == 0) {
      best_value = observer_biobj_get_matching_double_value(self->best_values_matrix[i], key,
          self->best_values_count, 0, 1, error_value);
      if (coco_doubles_almost_equal(best_value, error_value, error_accuracy) == 0) {
        coco_error("observer_biobj_read_best_value(): could not find %s in best value file", key);
        return 0; /* Never reached */
      } else
        return best_value;
    }
  }

  coco_error("observer_biobj_read_best_value(): unexpected exception");
  return 0; /* Never reached */

}

/**
 * Frees the memory of the given biobjective observer.
 */
static void observer_biobj_free(void *stuff) {

  observer_biobj_t *data;
  size_t i;

  assert(stuff != NULL);
  data = stuff;

  if (data->compute_indicators != 0) {
    for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
      observer_biobj_free_matrix_of_strings(data->best_values_matrix[i],
          data->best_values_count, 2);
    }
  }

}

/**
 * Initializes the biobjective observer. Possible options:
 * - log_nondominated : none (don't log nondominated solutions)
 * - log_nondominated : final (log only the final nondominated solutions; default value)
 * - log_nondominated : all (log every solution that is nondominated at creation time)
 * - include_decision_variables : 0 / 1 (whether to include decision variables when logging nondominated solutions;
 * default value is 0)
 * - compute_indicators : 0 / 1 (whether to compute and output performance indicators; default value is 1)
 * - produce_all_data: 0 / 1 (whether to produce all data; if set to 1, overwrites other options and is equivalent to
 * setting log_nondominated to all, include_decision_variables to 1 and compute_indicators to 1; if set to 0, it
 * does not change the values of other options; default value is 0)
 */
static void observer_biobj(coco_observer_t *self, const char *options) {

  observer_biobj_t *data;
  char string_value[COCO_PATH_MAX];
  char *file_name;
  FILE *file;
  size_t i;

  data = coco_allocate_memory(sizeof(*data));

  data->log_mode = FINAL;
  if (coco_options_read_string(options, "log_nondominated", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      data->log_mode = NONE;
    else if (strcmp(string_value, "all") == 0)
      data->log_mode = ALL;
  }

  if (coco_options_read_int(options, "include_decision_variables", &(data->include_decision_variables)) == 0)
    data->include_decision_variables = 0;

  if (coco_options_read_int(options, "compute_indicators", &(data->compute_indicators)) == 0)
    data->compute_indicators = 1;

  if (coco_options_read_int(options, "produce_all_data", &(data->produce_all_data)) == 0)
    data->produce_all_data = 0;

  if (data->produce_all_data) {
    data->include_decision_variables = 1;
    data->compute_indicators = 1;
    data->log_mode = ALL;
  }

  if (data->compute_indicators) {
    for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
      /* Load the data from the file with best values */
      file_name = coco_strdupf("best_values_%s.txt", OBSERVER_BIOBJ_INDICATORS[i]);
      file = fopen(file_name, "r");
      if (file == NULL) {
        coco_error("observer_biobj() failed to open file '%s'.", file_name);
        return; /* Never reached */
      }
      data->best_values_count = observer_biobj_get_number_of_lines_in_file(file);
      data->best_values_matrix[i] = observer_biobj_get_string_pairs_from_file(file,
          data->best_values_count, OBSERVER_BIOBJ_MAX_STR_LENGTH);
      fclose(file);
    }
    data->previous_function = -1;
  }

  self->logger_initialize_function = logger_biobj;
  self->data_free_function = observer_biobj_free;
  self->data = data;

  if ((data->log_mode == NONE) && (!data->compute_indicators)) {
    /* No logging required */
    self->is_active = 0;
  }
}
