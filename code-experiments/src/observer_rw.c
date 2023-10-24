/**
 * @file observer_rw.c
 * @brief Implementation of an observer for real-world problems.
 */

#include "coco.h"
#include "coco_internal.h"

#include "coco_utilities.c"

/** @brief Enum for denoting when the decision variables and constraints are logged. */
typedef enum {
  LOG_NEVER, LOG_LOW_DIM, LOG_ALWAYS
} observer_rw_log_e;

/**
 * @brief The real-world observer data type.
 */
typedef struct {
  observer_rw_log_e log_vars_mode;   /**< @brief When the decision variables are logged. */
  observer_rw_log_e log_cons_mode;   /**< @brief When the constraints are logged. */
  size_t low_dim_vars;               /**< @brief "Low dimension" for decision variables. */
  size_t low_dim_cons;               /**< @brief "Low dimension" for constraints. */
  int log_only_better;               /**< @brief Whether to log only solutions that are better than previous
                                                 ones (only for the single-objective problems). */
  int log_time;                      /**< @brief Whether to log time. */
} observer_rw_data_t;

static coco_problem_t *logger_rw(coco_observer_t *observer, coco_problem_t *problem);
static void logger_rw_free(void *logger);

/**
 * @brief Initializes the observer for real-world problems.
 *
 * Possible options:
 *
 * - "log_variables: STRING" determines whether the decision variables are to be logged. STRING can take on
 * the values "none" (don't output decision variables), "low_dim"(output decision variables only for
 * dimensions lower or equal to low_dim_vars) and "all" (output all decision variables). The default value
 * is "all".
 *
 * - "log_constraints: STRING" determines whether the constraints are to be logged. STRING can take on the
 * values "none" (don't output constraints), "low_dim"(output constraints only for dimensions lower or equal
 * to low_dim_cons) and "all" (output all constraints). The default value is "all".
 *
 * - "low_dim_vars: VALUE" determines the value used to define "low_dim" for decision variables. The default
 * value is 10.
 *
 * - "low_dim_cons: VALUE" determines the value used to define "low_dim" for constraints. The default value
 * is 10.
 *
 * - "log_only_better: 0/1" determines whether all solutions are logged (0, default) or only the ones that
 * are better than previous ones (1). This is applicable only for the single-objective problems, where the
 * default value is 0. For multi-objective problems, all solutions are always logged.
 *
 * - "log_time: 0/1" determines whether the time needed to evaluate each solution is logged (0) or not (1).
 * The default value is 0.
 */
static void observer_rw(coco_observer_t *observer, const char *options, coco_option_keys_t **option_keys) {

  observer_rw_data_t *observer_data;
  char string_value[COCO_PATH_MAX + 1];

  /* Sets the valid keys for rw observer options
   * IMPORTANT: This list should be up-to-date with the code and the documentation */
  const char *known_keys[] = { "log_variables", "log_constraints", "low_dim_vars", "low_dim_cons",
      "log_only_better", "log_time" };
  *option_keys = coco_option_keys_allocate(sizeof(known_keys) / sizeof(char *), known_keys);

  observer_data = (observer_rw_data_t *) coco_allocate_memory(sizeof(*observer_data));

  observer_data->log_vars_mode = LOG_ALWAYS;
  if (coco_options_read_string(options, "log_variables", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      observer_data->log_vars_mode = LOG_NEVER;
    else if (strcmp(string_value, "all") == 0)
      observer_data->log_vars_mode = LOG_ALWAYS;
    else if (strcmp(string_value, "low_dim") == 0)
      observer_data->log_vars_mode = LOG_LOW_DIM;
  }

  observer_data->log_cons_mode = LOG_ALWAYS;
  if (coco_options_read_string(options, "log_constraints", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      observer_data->log_cons_mode = LOG_NEVER;
    else if (strcmp(string_value, "all") == 0)
      observer_data->log_cons_mode = LOG_ALWAYS;
    else if (strcmp(string_value, "low_dim") == 0)
      observer_data->log_cons_mode = LOG_LOW_DIM;
  }

  if (coco_options_read_size_t(options, "low_dim_vars", &(observer_data->low_dim_vars)) == 0)
    observer_data->low_dim_vars = 10;

  if (coco_options_read_size_t(options, "low_dim_cons", &(observer_data->low_dim_cons)) == 0)
    observer_data->low_dim_cons = 10;

  if (coco_options_read_int(options, "log_only_better", &(observer_data->log_only_better)) == 0)
    observer_data->log_only_better = 0;

  if (coco_options_read_int(options, "log_time", &(observer_data->log_time)) == 0)
    observer_data->log_time = 0;

  observer->logger_allocate_function = logger_rw;
  observer->logger_free_function = logger_rw_free;
  observer->restart_function = NULL;
  observer->data_free_function = NULL;
  observer->data = observer_data;
}
