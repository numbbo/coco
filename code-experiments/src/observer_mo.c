#include "coco.h"
#include "coco_internal.h"

#include "coco_utilities.c"

coco_problem_t *deprecated__logger_mo(coco_problem_t *problem, const char *options);
coco_problem_t *logger_mo(coco_observer_t *observer, coco_problem_t *problem);

/* Logging nondominated solutions mode */
typedef enum {NONE, FINAL, ALL} observer_mo_log_nondom_e;

/* Data for the multiobjective observer */
typedef struct {
  observer_mo_log_nondom_e log_mode;

  int include_decision_variables;
  int compute_indicators;
  int produce_all_data;

} observer_mo_data_t;

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

  self->logger_initialize_function = logger_mo;
  self->observer_free_function = NULL; /* We don't need a freeing method since all data fields are basic */
  self->data = data;
}

/**
 * Multiobjective observer. See the multiobjective logger (function logger_mo(problem, options))
 * for more information.
 */
static coco_problem_t *deprecated__observer_mo(coco_problem_t *problem, const char *options) {

  /* The information to be logged at each step is defined in the function
   * 'private_logger_mo_evaluate' in the file 'logger_mo.c' */
  problem = deprecated__logger_mo(problem, options);

  return problem;
}
