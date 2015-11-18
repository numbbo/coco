#include "coco.h"
#include "coco_utilities.c"
#include "logger_nondominated.c"

/**
 * Multiobjective observer for logging all nondominated solutions found when
 * a new solution is generated and evaluated.
 */
static coco_problem_t *observer_mo_toy(coco_problem_t *problem, const char *options) {
  /* Calculate target levels for first hitting times */
  static const size_t max_size_of_archive = 100000;
  char base_path[COCO_PATH_MAX] = { 0 };
  char filename[COCO_PATH_MAX] = { 0 };
  coco_join_path(base_path, sizeof(base_path), options, "log_nondominated_solutions",
      coco_problem_get_id(problem), NULL);
  if (coco_path_exists(base_path)) {
    /* TODO: Handle this differently - coco_create_unique_path() function is available now! */
    coco_error("Result directory exists.");
    return NULL; /* never reached */
  }
  coco_create_path(base_path);
  coco_join_path(filename, sizeof(filename), base_path, "nondominated_at_birth.txt", NULL);
  problem = logger_nondominated(problem, max_size_of_archive, filename);
  /* To control which information to be logged at each func eval, modify
   * the function 'lht_evaluate_function' in the file 'log_hitting_times.c' */
  return problem;
}

