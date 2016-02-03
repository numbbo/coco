/**
 * @file observer_bbob.c
 * @brief Implementation of the bbob observer.
 */

#include "coco.h"
#include "coco_utilities.c"

static coco_problem_t *logger_bbob(coco_observer_t *observer, coco_problem_t *problem);

/**
 * @brief The bbob observer data type.
 */
typedef struct {
  /* TODO: Can be used to store variables that need to be accessible during one run (i.e. for multiple
   * problems). For example, the following global variables from logger_bbob.c could be stored here: */
  size_t current_dim;
  size_t current_fun_id;
  /* ... and others */
} observer_bbob_data_t;

/**
 * @brief Initializes the bbob observer.
 */
static void observer_bbob(coco_observer_t *observer, const char *options) {

  observer->logger_initialize_function = logger_bbob;
  observer->data_free_function = NULL;
  observer->data = NULL;

  (void)options; /* To silence the compiler */
}
