#include "coco.h"
#include "coco_utilities.c"
#include "logger_mo.c"

/**
 * Multiobjective observer. See the multiobjective logger (function logger_mo(problem, options))
 * for more information.
 */
static coco_problem_t *observer_mo(coco_problem_t *problem, const char *options) {

  /* The information to be logged at each step is defined in the function
   * 'private_logger_mo_evaluate' in the file 'logger_mo.c' */
  problem = logger_mo(problem, options);

  return problem;
}
