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
  size_t bbob_nbpts_nbevals; /**< @brief nb fun eval triggers */
  size_t bbob_nbpts_fval;    /**< @brief f value difference to the optimal triggers */
} observer_bbob_data_t;

/**
 * @brief Initializes the bbob observer.
 *
 * Possible options:
 * - bbob_nbpts_nbevals: nb fun eval triggers are at 10**(i/bbob_nbpts_nbevals) (the default value in bbob is 20 )
 * - bbob_nbpts_fval: f value difference to the optimal triggers are at 10**(i/bbob_nbpts_fval)(the default value in bbob is 5 )
 */
static void observer_bbob(coco_observer_t *observer, const char *options) {
  
  observer_bbob_data_t *observer_bbob;
  
  observer_bbob = coco_allocate_memory(sizeof(*observer_bbob));  

  if ((coco_options_read_size_t(options, "nbpts_nbevals", &(observer_bbob->bbob_nbpts_nbevals)) == 0)) {
    observer_bbob->bbob_nbpts_nbevals = 20;
  }
  if ((coco_options_read_size_t(options, "nbpts_fval", &(observer_bbob->bbob_nbpts_fval)) == 0)) {
    observer_bbob->bbob_nbpts_fval = 5;
  }

  observer->logger_initialize_function = logger_bbob;
  observer->data_free_function = NULL;
  observer->data = observer_bbob;
}
