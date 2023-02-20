/**
 * @file observer_sbox_cost.c
 * @brief Implementation of the sbox_cost observer.
 */

#include "coco.h"
#include "coco_utilities.c"

static coco_problem_t *logger_sbox_cost(coco_observer_t *observer, coco_problem_t *problem);
static void logger_sbox_cost_free(void *logger);

/**
 * @brief The sbox_cost observer data type.
 */
typedef struct {
  /* TODO: Can be used to store variables that need to be accessible during one run (i.e. for multiple
   * problems). For example, the following global variables from logger_sbox_cost.c could be stored here: */
  size_t current_dim;
  size_t current_fun_id;
  /* ... and others */
} observer_sbox_cost_data_t;

/**
 * @brief Initializes the sbox_cost observer.
 */
static void observer_sbox_cost(coco_observer_t *observer, const char *options, coco_option_keys_t **option_keys) {

  observer->logger_allocate_function = logger_sbox_cost;
  observer->logger_free_function = logger_sbox_cost_free;
  observer->data_free_function = NULL;
  observer->data = NULL;

  *option_keys = NULL;

  (void) options; /* To silence the compiler */
}
