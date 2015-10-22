#include "coco_platform.h"

#include <string.h>
#include <stdio.h>

#include "coco.h"

#include "toy_suit.c"
#include "toy_observer.c"

#include "bbob2009_suite.c"
#include "bbob2009_observer.c"

#include "mo_suite_first_attempt.c"

#include "biobjective_suite_300.c"
#include "biobjective_observer.c"

/**
 * A new benchmark suite must provide a function that returns a
 * coco_problem or NULL when given an problem_index as input.
 *
 * The file containing this (static) function must be included above
 * and the function call must be added to coco_get_problem below.
 * 
 * If the benchmark does not have continuous problem indices starting with,
 * zero, additional functionality must also be added to coco_next_problem_index
 * (it should done in any case for efficiency).
 *
 * To construct a benchmark suite, useful tools are coco_transformed...
 * coco_stacked..., bbob2009_problem() and the various existing base
 * functions and transformations like shift_variables...
 */

/** return next problem_index or -1
 */
long coco_next_problem_index(const char *problem_suite,
                            long problem_index,
                            const char *select_options) {
  coco_problem_t *problem; /* to check validity */
  long last_index = -1;
  
  /* code specific to known benchmark suites */
  /* for efficiency reasons, each test suit should define
   * at least its last_index here */
  if (0 == strcmp(problem_suite, "bbob2009")) {
    /* without selection_options: last_index = 2159; */
    return bbob2009_next_problem_index(problem_index, select_options);
  }

  /** generic implementation:
   *   first index == 0,
   *   ++index until index > max_index or problem(index) == NULL
   **/
  
  if (problem_index < 0)
    problem_index = -1;
    
  ++problem_index;
  if (last_index >= 0) {
    if (problem_index <= last_index)
      return problem_index;
    else
      return -1;
  }
  
  /* last resort: last_index is not known */
  problem = coco_get_problem(problem_suite, problem_index);
  if (problem == NULL) {
    return -1;
  }
  coco_free_problem(problem);
  return problem_index;
}

/**
 * coco_suite_get_problem(problem_suite, problem_index):
 *
 * return the coco problem with index ${problem_index} from
 * suite ${problem_suite}. The problem must be de-allocated
 * using the function ${coco_free_problem}.
 *
 */
coco_problem_t *coco_get_problem(const char *problem_suite,
                                 const long problem_index) {
  if (0 == strcmp(problem_suite, "toy_suit")) {
    return toy_suit(problem_index);
  } else if (0 == strcmp(problem_suite, "bbob2009")) {
    return bbob2009_suite(problem_index);
  } else if (0 == strcmp(problem_suite, "mo_suite_first_attempt")) {
    return mo_suite_first_attempt(problem_index);
  } else if (0 == strcmp(problem_suite, "biobjective_suite_300")) {
    return biobjective_suite_300(problem_index);
  } else {
    coco_warning("Unknown problem suite.");
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
  } else if (0 == strcmp(observer, "mo_toy_observer")) {
    return mo_toy_observer(problem, options);
  }
  
  /* here each observer must have another entry */
  
  if (0 == strcmp(observer, "no_observer")) {
    return problem;
  } else if (strlen(observer) == 0) {
    coco_warning("Empty observer '' has no effect. To prevent this warning use 'no_observer' instead");
    return problem;
  } else {
    /* not so clear whether an error is better, depends on the usecase */
    coco_warning(observer);
    coco_warning("is an unkown observer which has no effect (the reason might just be a typo)");
    return problem;
  }
  coco_error("Unknown observer.");
  return NULL; /* Never reached */
}

#if 1
/* coco_benchmark(problem_suite, observer, options, optimizer):
 *
 * Benchmark a solver ${optimizer} with a testbed ${problem_suite}
 * using the data logger ${observer} to write data. 
 */
void coco_benchmark(const char *problem_suite, const char *observer,
                    const char *options, coco_optimizer_t optimizer) {
  int problem_index;
  coco_problem_t *problem;
  for (problem_index = 0;; ++problem_index) {
    problem = coco_get_problem(problem_suite, problem_index);
    if (NULL == problem)
      break;
    problem = coco_observe_problem(observer, problem, options); /* should remain invisible to the user*/
    optimizer(problem);
    /* Free problem after optimization. */
    coco_free_problem(problem);
  }
}

#else
/** "improved" interface for coco_benchmark: is it worth-while to have suite-options on the C-level? 
 */
void coco_benchmark(const char *problem_suite, const char *problem_suite_options,
                     const char *observer, const char *observer_options,
                     coco_optimizer_t optimizer) {
  int problem_index;
  int is_instance;
  coco_problem_t *problem;
  char buf[222]; /* TODO: this is ugly, how to improve? The new implementation of coco_warning makes this obsolete */
  for (problem_index = -1; ; ) {
    problem_index = coco_next_problem_index(problem_suite, problem_suite_options, problem_index); 
    if (problem_index < 0)
      break;
    problem = coco_get_problem(problem_suite, problem_index);
    if (problem == NULL)
      snprintf(buf, 221, "problem index %d not found in problem suit %s (this is probably a bug)",
               problem_index, problem_suite); 
      coco_warning(buf);
      break;
    problem = coco_observe_problem(observer, problem, observer_options); /* should remain invisible to the user*/
    optimizer(problem);
    coco_free_problem(problem);
  }
}
#endif
