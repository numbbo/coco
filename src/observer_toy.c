#include "coco.h"
#include "coco_utilities.c"
#include "logger_target_hits.c"

static coco_problem_t *observer_toy(coco_problem_t *problem, const char *options) {
  size_t i;
  static const size_t number_of_targets = 20;
  double targets[20];
  char base_path[COCO_PATH_MAX] = { 0 };
  char filename[COCO_PATH_MAX] = { 0 };

  /* Calculate target levels: */
  for (i = number_of_targets; i > 0; --i) {
    targets[i - 1] = pow(10.0, (double) (long) (number_of_targets - i) - 9.0);
  }

  coco_join_path(base_path, sizeof(base_path), options, "toy_so", coco_get_problem_id(problem), NULL);
  if (coco_path_exists(base_path)) {
    coco_error("Result directory exists.");
    return NULL; /* never reached */
  }
  coco_create_path(base_path);
  coco_join_path(filename, sizeof(filename), base_path, "first_hitting_times.txt", NULL);
  problem = logger_target_hits(problem, targets, number_of_targets, filename);
  return problem;
}
