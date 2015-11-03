#include "coco.h"
#include "coco_utilities.c"
#include "logger_mo.c"

/**
 * Multiobjective observer for logging all sorts of things... TODO: Improve description!
 */
static coco_problem_t *observer_mo(coco_problem_t *problem, const char *options) {

  char base_path[COCO_PATH_MAX] = { 0 };
  char filename[COCO_PATH_MAX] = { 0 };

  coco_join_path(base_path, sizeof(base_path), options, coco_problem_get_id(problem), NULL);
  if (coco_path_exists(base_path)) {
    coco_error("Result directory exists.");
    return NULL; /* never reached */
  }
  coco_create_path(base_path);
  coco_join_path(filename, sizeof(filename), base_path, "nondominated_at_birth.txt", NULL);

  problem = logger_mo(problem, filename);
  /* The information to be logged at each step is defined in the function
   * 'private_logger_mo_evaluate' in the file 'logger_mo.c' */

  return problem;
}

