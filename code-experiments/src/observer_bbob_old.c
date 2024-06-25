/**
 * @file observer_bbob_old.c
 * @brief Old implementation of the bbob observer.
 */

#include "coco.h"
#include "coco_utilities.c"

static coco_problem_t *logger_bbob_old(coco_observer_t *observer, coco_problem_t *problem);
static void logger_bbob_old_free(void *logger);

/**
 * @brief The bbob_old observer data type.
 */
typedef struct {
  /* TODO: Can be used to store variables that need to be accessible during one run (i.e. for multiple
   * problems). For example, the following global variables from logger_bbob_old.c could be stored here: */
  size_t current_dim;
  size_t current_fun_id;
  /* ... and others */
} observer_bbob_old_data_t;

/**
 * @brief Initializes the bbob_old observer.
 */
static void observer_bbob_old(coco_observer_t *observer, const char *options, coco_option_keys_t **option_keys) {

  observer->logger_allocate_function = logger_bbob_old;
  observer->logger_free_function = logger_bbob_old_free;
  observer->restart_function = NULL;
  observer->data_free_function = NULL;
  observer->data = NULL;

  *option_keys = NULL;

  (void) options; /* To silence the compiler */
}
