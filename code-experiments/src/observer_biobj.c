#include "coco.h"
#include "coco_internal.h"

#include "coco_utilities.c"
#include "mo_generics.c"

/* List of implemented indicators */
#define OBSERVER_BIOBJ_NUMBER_OF_INDICATORS 1
const char *OBSERVER_BIOBJ_INDICATORS[OBSERVER_BIOBJ_NUMBER_OF_INDICATORS] = { "hyp" };

/* Logging nondominated solutions mode */
typedef enum {
  NONE, FINAL, ALL
} observer_biobj_log_nondom_e;

/* Logging variables mode */
typedef enum {
  NEVER, LOW_DIM, ALWAYS
} observer_biobj_log_vars_e;

/* Data for the biobjective observer */
typedef struct {
  observer_biobj_log_nondom_e log_nondom_mode;
  observer_biobj_log_vars_e log_vars_mode;

  int compute_indicators;
  int produce_all_data;

  /* Information on the previous logged problem */
  long previous_function;

} observer_biobj_t;

static coco_problem_t *logger_biobj(coco_observer_t *observer, coco_problem_t *problem);

/**
 * Initializes the biobjective observer. Possible options:
 * - log_nondominated : none (don't log nondominated solutions)
 * - log_nondominated : final (log only the final nondominated solutions)
 * - log_nondominated : all (log every solution that is nondominated at creation time; default value)
 * - log_decision_variables : none (don't output decision variables)
 * - log_decision_variables : log_dim (output decision variables only for dimensions lower or equal to 5; default value)
 * - log_decision_variables : all (output all decision variables)
 * - compute_indicators : 0 / 1 (whether to compute and output performance indicators; default value is 1)
 * - produce_all_data: 0 / 1 (whether to produce all data; if set to 1, overwrites other options and is equivalent to
 * setting log_nondominated to all, log_decision_variables to log_dim and compute_indicators to 1; if set to 0, it
 * does not change the values of other options; default value is 0)
 */
static void observer_biobj(coco_observer_t *observer, const char *options) {

  observer_biobj_t *observer_biobj;
  char string_value[COCO_PATH_MAX];

  observer_biobj = coco_allocate_memory(sizeof(*observer_biobj));

  observer_biobj->log_nondom_mode = ALL;
  if (coco_options_read_string(options, "log_nondominated", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      observer_biobj->log_nondom_mode = NONE;
    else if (strcmp(string_value, "final") == 0)
      observer_biobj->log_nondom_mode = FINAL;
  }

  observer_biobj->log_vars_mode = LOW_DIM;
  if (coco_options_read_string(options, "log_decision_variables", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      observer_biobj->log_vars_mode = NEVER;
    else if (strcmp(string_value, "all") == 0)
      observer_biobj->log_vars_mode = ALWAYS;
  }

  if (coco_options_read_int(options, "compute_indicators", &(observer_biobj->compute_indicators)) == 0)
    observer_biobj->compute_indicators = 1;

  if (coco_options_read_int(options, "produce_all_data", &(observer_biobj->produce_all_data)) == 0)
    observer_biobj->produce_all_data = 0;

  if (observer_biobj->produce_all_data) {
    observer_biobj->log_vars_mode = LOW_DIM;
    observer_biobj->compute_indicators = 1;
    observer_biobj->log_nondom_mode = ALL;
  }

  if (observer_biobj->compute_indicators) {
    observer_biobj->previous_function = -1;
  }

  observer->logger_initialize_function = logger_biobj;
  observer->data_free_function = NULL;
  observer->data = observer_biobj;

  if ((observer_biobj->log_nondom_mode == NONE) && (!observer_biobj->compute_indicators)) {
    /* No logging required */
    observer->is_active = 0;
  }
}
