/**
 * @file observer_biobj.c
 * @brief Implementation of the bbob-biobj observer.
 */

#include "coco.h"
#include "coco_internal.h"

#include "coco_utilities.c"
#include "mo_generics.c"

/** @brief Number of implemented indicators */
#define OBSERVER_BIOBJ_NUMBER_OF_INDICATORS 1

/** @brief Names of implemented indicators */
const char *observer_biobj_indicators[OBSERVER_BIOBJ_NUMBER_OF_INDICATORS] = { "hyp" };

/** @brief Enum for denoting the way in which the nondominated solutions are logged. */
typedef enum {
  LOG_NONDOM_NONE, LOG_NONDOM_FINAL, LOG_NONDOM_ALL
} observer_biobj_log_nondom_e;

/** @brief Enum for denoting the when the decision variables are logged. */
typedef enum {
  LOG_VARS_NEVER, LOG_VARS_LOW_DIM, LOG_VARS_ALWAYS
} observer_biobj_log_vars_e;

/**
 * @brief The bbob-biobj observer data type.
 */
typedef struct {
  observer_biobj_log_nondom_e log_nondom_mode; /**< @brief How the nondominated solutions are logged. */
  observer_biobj_log_vars_e log_vars_mode;     /**< @brief When the decision variables are logged. */

  int compute_indicators;                      /**< @brief Whether to compute indicators. */
  int produce_all_data;                        /**< @brief Whether to produce all data. */

  long previous_function;                      /**< @brief Information on the previous logged problem. */

} observer_biobj_t;

static coco_problem_t *logger_biobj(coco_observer_t *observer, coco_problem_t *problem);

/**
 * @brief Initializes the bi-objective observer.
 *
 * Possible options:
 * - log_nondominated : none (don't log nondominated solutions)
 * - log_nondominated : final (log only the final nondominated solutions)
 * - log_nondominated : all (log every solution that is nondominated at creation time; default value)
 * - log_decision_variables : none (don't output decision variables)
 * - log_decision_variables : log_dim (output decision variables only for dimensions lower or equal to 5;
 * default value)
 * - log_decision_variables : all (output all decision variables)
 * - compute_indicators : 0 / 1 (whether to compute and output performance indicators; default value is 1)
 * - produce_all_data: 0 / 1 (whether to produce all data; if set to 1, overwrites other options and is
 * equivalent to setting log_nondominated to all, log_decision_variables to log_dim and compute_indicators
 * to 1; if set to 0, it does not change the values of other options; default value is 0)
 */
static void observer_biobj(coco_observer_t *observer, const char *options) {

  observer_biobj_t *observer_biobj;
  char string_value[COCO_PATH_MAX];

  observer_biobj = coco_allocate_memory(sizeof(*observer_biobj));

  observer_biobj->log_nondom_mode = LOG_NONDOM_ALL;
  if (coco_options_read_string(options, "log_nondominated", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      observer_biobj->log_nondom_mode = LOG_NONDOM_NONE;
    else if (strcmp(string_value, "final") == 0)
      observer_biobj->log_nondom_mode = LOG_NONDOM_FINAL;
  }

  observer_biobj->log_vars_mode = LOG_VARS_LOW_DIM;
  if (coco_options_read_string(options, "log_decision_variables", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      observer_biobj->log_vars_mode = LOG_VARS_NEVER;
    else if (strcmp(string_value, "all") == 0)
      observer_biobj->log_vars_mode = LOG_VARS_ALWAYS;
  }

  if (coco_options_read_int(options, "compute_indicators", &(observer_biobj->compute_indicators)) == 0)
    observer_biobj->compute_indicators = 1;

  if (coco_options_read_int(options, "produce_all_data", &(observer_biobj->produce_all_data)) == 0)
    observer_biobj->produce_all_data = 0;

  if (observer_biobj->produce_all_data) {
    observer_biobj->log_vars_mode = LOG_VARS_LOW_DIM;
    observer_biobj->compute_indicators = 1;
    observer_biobj->log_nondom_mode = LOG_NONDOM_ALL;
  }

  if (observer_biobj->compute_indicators) {
    observer_biobj->previous_function = -1;
  }

  observer->logger_initialize_function = logger_biobj;
  observer->data_free_function = NULL;
  observer->data = observer_biobj;

  if ((observer_biobj->log_nondom_mode == LOG_NONDOM_NONE) && (!observer_biobj->compute_indicators)) {
    /* No logging required */
    observer->is_active = 0;
  }
}
