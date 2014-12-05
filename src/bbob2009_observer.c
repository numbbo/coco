#include "coco.h"

#include "bbob2009_logger.c"

coco_problem_t *bbob2009_observer(coco_problem_t *problem,
                                  const char *options) {

  coco_create_path(options);
  problem = bbob2009_logger(problem, options);
  return problem;
}
