#include "coco.h"
#include "coco_internal.h"

#include "coco_utilities.c"
#include "mo_generics.c"

coco_problem_t *logger_mo(coco_observer_t *self, coco_problem_t *problem);

/* List of implemented indicators */
#define OBSERVER_MO_NUMBER_OF_INDICATORS 1
const char *OBSERVER_MO_INDICATORS[OBSERVER_MO_NUMBER_OF_INDICATORS] = { "hypervolume" };
const size_t OBSERVER_MO_MAX_STR_LENGTH = 100;

/* Logging nondominated solutions mode */
typedef enum {
  NONE, FINAL, ALL
} observer_mo_log_nondom_e;

/* Data for the multiobjective observer */
typedef struct {
  observer_mo_log_nondom_e log_mode;

  int include_decision_variables;
  int compute_indicators;
  int produce_all_data;

  char ***reference_value_matrix[OBSERVER_MO_NUMBER_OF_INDICATORS];
  size_t reference_values_count;

} observer_mo_data_t;

static void observer_mo_free(coco_observer_t *observer) {

  observer_mo_data_t *observer_mo;
  size_t i;

  assert(observer != NULL);
  observer_mo = (observer_mo_data_t *) observer;

  if (observer_mo->compute_indicators != 0) {
    for (i = 0; i < OBSERVER_MO_NUMBER_OF_INDICATORS; i++) {
      mo_free_matrix_of_strings(observer_mo->reference_value_matrix[i], observer_mo->reference_values_count,
          2);
    }
  }
}

/**
 * Initializes the multiobjective observer. Possible options:
 * - log_nondominated : none (don't log nondominated solutions)
 * - log_nondominated : final (log only the final nondominated solutions; default value)
 * - log_nondominated : all (log every solution that is nondominated at creation time)
 * - include_decision_variables : 0 / 1 (whether to include decision variables when logging nondominated solutions;
 * default value is 0)
 * - compute_indicators : 0 / 1 (whether to compute performance indicators; default value is 1)
 * - produce_all_data: 0 / 1 (whether to produce all data; if set to 1, overwrites other options and is equivalent to
 * setting log_nondominated to all, include_decision_variables to 1 and compute_indicators to 1; if set to 0, it
 * does not change the values of other options; default value is 0)
 */
static void observer_mo(coco_observer_t *self, const char *options) {

  observer_mo_data_t *data;
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

  if ((data->log_mode == NONE) && (!data->compute_indicators)) {
    /* No logging required, return NULL */
    return;
  }

  if (data->compute_indicators) {
    for (i = 0; i < OBSERVER_MO_NUMBER_OF_INDICATORS; i++) {
      /* Load the data from the file with reference values */
      file_name = coco_strdupf("reference_values_%s.txt", OBSERVER_MO_INDICATORS[i]);
      file = fopen(file_name, "r");
      if (file == NULL) {
        coco_error("observer_mo() failed to open file '%s'.", file_name);
        return; /* Never reached */
      }
      data->reference_values_count = mo_get_number_of_lines_in_file(file);
      data->reference_value_matrix[i] = mo_get_string_pairs_from_file(file, data->reference_values_count,
          OBSERVER_MO_MAX_STR_LENGTH);
      fclose(file);
    }
  }

  self->logger_initialize_function = logger_mo;
  self->observer_free_function = observer_mo_free;
  self->data = data;
}

/* Returns the reference value for indicator_name matching the given key if the key is found, and raises an
 * error otherwise.  */
static double observer_mo_get_reference_value(const observer_mo_data_t *self,
                                              const char *indicator_name,
                                              const char *key) {

  size_t i;
  double reference_value;
  double error_value = -1;
  double error_accuracy = 1e-8;

  for (i = 0; i < OBSERVER_MO_NUMBER_OF_INDICATORS; i++) {
    if (strcmp(OBSERVER_MO_INDICATORS[i], indicator_name) == 0) {
      reference_value = mo_get_matching_double_value(self->reference_value_matrix[i], key, self->reference_values_count,
          0, 1, error_value);
      if (coco_doubles_almost_equal(reference_value, error_value, error_accuracy) == 0) {
        coco_error("observer_mo_get_reference_value(): could not find %s in reference file", key);
        return 0; /* Never reached */
      }
      else
        return reference_value;
    }
  }

  coco_error("observer_mo_get_reference_value(): unexpected exception");
  return 0; /* Never reached */

}
