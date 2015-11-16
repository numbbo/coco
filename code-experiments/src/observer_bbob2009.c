#include "coco.h"

#include "logger_bbob2009.c"

static coco_problem_t *observer_bbob2009(coco_problem_t *problem, const char *options) {
  if (problem == NULL)
    return problem;
  /* TODO: " */
  coco_create_path(options);
  problem = logger_bbob2009(problem, options);
  return problem;
}
