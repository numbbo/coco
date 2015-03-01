#include "coco.h"

#include "bbob2009_logger.c"

static coco_problem_t *bbob2009_observer(coco_problem_t *problem,
                                  const char *options) {
  if (problem == NULL)
    return problem;
  /* TODO: parse options that look like "folder:foo; verbosity:bar" */
  coco_create_path(options);
  problem = bbob2009_logger(problem, options);
  return problem;
}
