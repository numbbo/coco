#include "coco_platform.h"

#include <string.h>
#include <stdio.h>

#include "coco.h"

#include "toy_suit.c"
#include "toy_observer.c"

#include "bbob2009_suit.c"
#include "bbob2009_observer.c"

/** return next problem_index or -1
 */
int coco_next_problem_index(const char *problem_suit,
                     const char *select_options,
                     const int problem_index) {
  if (0 == strcmp(problem_suit, "bbob2009")) {
    /* TODO: this is not (yet) implemented */
    return bbob2009_next_problem_index(select_options, problem_index);
  }
  return problem_index + 1;
}

coco_problem_t *coco_get_problem(const char *problem_suit,
                                 const int problem_index) {
  if (0 == strcmp(problem_suit, "toy_suit")) {
    return toy_suit(problem_index);
  } else if (0 == strcmp(problem_suit, "bbob2009")) {
    return bbob2009_suit(problem_index);
  } else {
    coco_warning("Unknown problem suit.");
    return NULL;
  }
}

coco_problem_t *coco_observe_problem(const char *observer,
                                     coco_problem_t *problem,
                                     const char *options) {
  if (problem == NULL) {
    coco_warning("Trying to observe a NULL problem has no effect.");
    return problem;
  }
  if (0 == strcmp(observer, "toy_observer")) {
    return toy_observer(problem, options);
  } else if (0 == strcmp(observer, "bbob2009_observer")) {
    return bbob2009_observer(problem, options);
  } else if (0 == strcmp(observer, "") ||
             0 == strcmp(observer, "no_observer")) {
    return problem;
  } else {
    coco_error("Unknown observer.");
    return NULL; /* Never reached */
  }
}

#if 1
void coco_benchmark(const char *problem_suit, const char *observer,
                    const char *options, coco_optimizer_t optimizer) {
  int problem_index;
  coco_problem_t *problem;
  for (problem_index = 0;; ++problem_index) {
    problem = coco_get_problem(problem_suit, problem_index);
    if (NULL == problem)
      break;
    problem = coco_observe_problem(observer, problem, options); /* should remain invisible to the user*/
    optimizer(problem);
    /* Free problem after optimization. */
    coco_free_problem(problem);
  }
}

#else
/** improved interface for coco_benchmark:
 */
void coco_benchmark(const char *problem_suit, const char *problem_suit_options,
                     const char *observer, const char *observer_options,
                     coco_optimizer_t optimizer) {
  int problem_index;
  int is_instance;
  coco_problem_t *problem;
  char buf[222]; 
  for (problem_index = -1; ; ) {
    problem_index = coco_next_problem_index(problem_suit, problem_suit_options, problem_index); 
    if (problem_index < 0)
      break;
    problem = coco_get_problem(problem_suit, problem_index);
    if (problem == NULL)
      snprintf(buf, 221, "problem index %d not found in problem suit %s (this is probably a bug)",
               problem_index, problem_suit); 
      coco_warning(buf);
      break;
    problem = coco_observe_problem(observer, problem, observer_options); /* should remain invisible to the user*/
    optimizer(problem);
    coco_free_problem(problem);
  }
}
#endif
