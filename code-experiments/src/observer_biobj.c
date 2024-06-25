/**
 * @file observer_biobj.c
 * @brief Implementation of the bbob-biobj observer.
 */

#include "coco.h"
#include "coco_internal.h"

#include "coco_utilities.c"
#include "mo_utilities.c"

/** @brief Enum for denoting the way in which the nondominated solutions are treated. */
typedef enum {
  LOG_NONDOM_NONE, LOG_NONDOM_FINAL, LOG_NONDOM_ALL, LOG_NONDOM_READ
} observer_biobj_log_nondom_e;

/** @brief Enum for denoting when the decision variables are logged. */
typedef enum {
  LOG_VARS_NEVER, LOG_VARS_LOW_DIM, LOG_VARS_ALWAYS
} observer_biobj_log_vars_e;

static coco_problem_t *logger_biobj(coco_observer_t *observer, coco_problem_t *problem);
static void logger_biobj_free(void *logger);
static void logger_biobj_signal_restart(coco_problem_t *problem);
static void logger_biobj_data_nullify_observer(void *logger_data);

/**
 * @brief The bbob-biobj observer data type.
 *
 * There is a cyclic reference between the biobj_logger and the biobj_observer (through the observer's
 * observed_problem and the logger's data, which points to the observer). This is needed to be able
 * to free both objects without problems.
 */
typedef struct {
  coco_problem_t *observed_problem;            /**< @brief Pointer to the observed problem (NULL if none is observed) */

  observer_biobj_log_nondom_e log_nondom_mode; /**< @brief Handling of the nondominated solutions. */
  observer_biobj_log_vars_e log_vars_mode;     /**< @brief When the decision variables are logged. */

  int compute_indicators;                      /**< @brief Whether to compute indicators. */
  int produce_all_data;                        /**< @brief Whether to produce all data. */

  long previous_function;                      /**< @brief Function of the previous logged problem. */
  long previous_dimension;                     /**< @brief Dimension of the previous logged problem */

} observer_biobj_data_t;

/**
 * @brief  Makes sure the observer_biobj_data_t object is not pointing to any problem.
 */
static void observer_biobj_data_free(void *stuff) {

  observer_biobj_data_t *data = (observer_biobj_data_t *) stuff;
  coco_problem_t *problem;

  coco_debug("Started observer_bbob_data_free()");

  /* Make sure that the observed problem's pointer to the observer points to NULL */
  if (data->observed_problem != NULL) {
    problem = (coco_problem_t *) data->observed_problem;
    if (problem->data != NULL) {
      logger_biobj_data_nullify_observer(coco_problem_transformed_get_data(problem));
    }
    data->observed_problem = NULL;
  }

  coco_debug("Ended   observer_bbob_data_free()");
}

/**
 * @brief Initializes the bi-objective observer.
 *
 * Possible options:
 *
 * - "log_nondominated: STRING" determines how the nondominated solutions are handled. STRING can take on the
 * values "none" (don't log nondominated solutions), "final" (log only the final nondominated solutions),
 * "all" (log every solution that is nondominated at creation time) and "read" (the nondominated solutions
 * are not logged, but are passed to the logger as input - this is a functionality needed in pre-processing
 * of the data). The default value is "all".
 *
 * - "log_decision_variables: STRING" determines whether the decision variables are to be logged in addition
 * to the objective variables in the output of nondominated solutions. STRING can take on the values "none"
 * (don't output decision variables), "low_dim"(output decision variables only for dimensions lower or equal
 * to 5) and "all" (output all decision variables). The default value is "low_dim".
 *
 * - "compute_indicators: VALUE" determines whether to compute and output performance indicators (1) or not
 * (0). The default value is 1.
 *
 * - "produce_all_data: VALUE" determines whether to produce all data required for the workshop. If set to 1,
 * it overwrites some other options and is equivalent to setting "log_nondominated: all",
 * "log_decision_variables: low_dim" and "compute_indicators: 1". If set to 0, it does not change the values
 * of the other options. The default value is 0.
 */
static void observer_biobj(coco_observer_t *observer, const char *options, coco_option_keys_t **option_keys) {

  observer_biobj_data_t *observer_data;
  char string_value[COCO_PATH_MAX + 1];

  /* Sets the valid keys for bbob-biobj observer options
   * IMPORTANT: This list should be up-to-date with the code and the documentation */
  const char *known_keys[] = { "log_nondominated", "log_decision_variables", "compute_indicators",
      "produce_all_data" };
  *option_keys = coco_option_keys_allocate(sizeof(known_keys) / sizeof(char *), known_keys);

  observer_data = (observer_biobj_data_t *) coco_allocate_memory(sizeof(*observer_data));

  observer_data->log_nondom_mode = LOG_NONDOM_ALL;
  if (coco_options_read_string(options, "log_nondominated", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      observer_data->log_nondom_mode = LOG_NONDOM_NONE;
    else if (strcmp(string_value, "final") == 0)
      observer_data->log_nondom_mode = LOG_NONDOM_FINAL;
    else if (strcmp(string_value, "all") == 0)
      observer_data->log_nondom_mode = LOG_NONDOM_ALL;
    else if (strcmp(string_value, "read") == 0)
      observer_data->log_nondom_mode = LOG_NONDOM_READ;
  }

  observer_data->log_vars_mode = LOG_VARS_LOW_DIM;
  if (coco_options_read_string(options, "log_decision_variables", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      observer_data->log_vars_mode = LOG_VARS_NEVER;
    else if (strcmp(string_value, "all") == 0)
      observer_data->log_vars_mode = LOG_VARS_ALWAYS;
    else if (strcmp(string_value, "low_dim") == 0)
      observer_data->log_vars_mode = LOG_VARS_LOW_DIM;
  }

  if (coco_options_read_int(options, "compute_indicators", &(observer_data->compute_indicators)) == 0)
    observer_data->compute_indicators = 1;

  if (coco_options_read_int(options, "produce_all_data", &(observer_data->produce_all_data)) == 0)
    observer_data->produce_all_data = 0;

  if (observer_data->produce_all_data) {
    observer_data->compute_indicators = 1;
    observer_data->log_nondom_mode = LOG_NONDOM_ALL;
    observer_data->log_vars_mode = LOG_VARS_LOW_DIM;
  }

  if (observer_data->compute_indicators) {
    observer_data->previous_function = -1;
    observer_data->previous_dimension = -1;
  }

  observer_data->observed_problem = NULL;

  observer->logger_allocate_function = logger_biobj;
  observer->logger_free_function = logger_biobj_free;
  observer->restart_function = logger_biobj_signal_restart;
  observer->data_free_function = observer_biobj_data_free;
  observer->data = observer_data;

  if ((observer_data->log_nondom_mode == LOG_NONDOM_NONE) && (!observer_data->compute_indicators)) {
    /* No logging required */
    observer->is_active = 0;
  }
}
