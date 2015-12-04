#include "coco_platform.h"

#include <string.h>
#include <stdio.h>

#include "coco.h"

#include "observer_bbob2009.c"
#include "suite_bbob2009.c"
#include "suite_biobj_300.c"
#include "suite_toy.c"

/**
 * A new benchmark suite must provide a function that returns a
 * coco_problem or NULL when given an problem_index as input.
 *
 * The file containing this (static) function must be included above
 * and the function call must be added to coco_suite_get_problem below.
 * 
 * If the benchmark does not have continuous problem indices starting with,
 * zero, additional functionality must also be added to coco_suite_get_next_problem_index
 * (it should done in any case for efficiency).
 *
 * To construct a benchmark suite, useful tools are coco_transformed...
 * coco_stacked..., bbob2009_problem() and the various existing base
 * functions and transformations like transform_vars_shift...
 */

/** return next problem_index or -1
 */
long coco_suite_get_next_problem_index(const char *problem_suite,
                                       long problem_index,
                                       const char *select_options) {
  coco_problem_t *problem; /* to check validity */
  long last_index = -1;

  /* code specific to known benchmark suites */
  /* for efficiency reasons, each test suite should define
   * at least its last_index here */
  if (0 == strcmp(problem_suite, "suite_bbob2009")) {
    /* without selection_options: last_index = 2159; */
    return suite_bbob2009_get_next_problem_index(problem_index, select_options);
  } else if (0 == strcmp(problem_suite, "suite_biobj_300")) {
    return suite_biobj_300_get_next_problem_index(problem_index, select_options);
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
  problem = coco_suite_get_problem(problem_suite, problem_index);
  if (problem == NULL) {
    return -1;
  }
  coco_problem_free(problem);
  return problem_index;
}

/**
 * coco_suite_get_problem(problem_suite, problem_index):
 *
 * return the coco problem with index ${problem_index} from
 * suite ${problem_suite}. The problem must be de-allocated
 * using the function ${coco_problem_free}.
 *
 */
coco_problem_t *coco_suite_get_problem(const char *problem_suite, const long problem_index) {
  if (0 == strcmp(problem_suite, "suite_toy")) {
    return suite_toy(problem_index);
  } else if (0 == strcmp(problem_suite, "suite_bbob2009")) {
    return suite_bbob2009(problem_index);
  } else if (0 == strcmp(problem_suite, "suite_biobj_300")) {
    return suite_biobj_300(problem_index);
  } else {
    coco_warning("Unknown problem suite.");
    return NULL;
  }
}

coco_problem_t *deprecated__coco_problem_add_observer(coco_problem_t *problem,
                                                      const char *observer_name,
                                                      const char *options) {
  if (problem == NULL) {
    coco_warning("Trying to observe a NULL problem has no effect.");
    return problem;
  }
  if (0 == strcmp(observer_name, "observer_toy")) {
    coco_error("Deprecated way of calling the toy observer.");
    return NULL; /* Never reached */
  } else if (0 == strcmp(observer_name, "observer_bbob2009")) {
    return deprecated__observer_bbob2009(problem, options);
  } else if (0 == strcmp(observer_name, "observer_biobj")) {
    coco_error("Deprecated way of calling the MO observer.");
    return NULL; /* Never reached */
  }

  /* here each observer must have another entry */

  if (0 == strcmp(observer_name, "no_observer")) {
    return problem;
  } else if (strlen(observer_name) == 0) {
    coco_warning("Empty observer '' has no effect. To prevent this warning use 'no_observer' instead");
    return problem;
  } else {
    /* not so clear whether an error is better, depends on the usecase */
    coco_warning(observer_name);
    coco_warning("is an unknown observer which has no effect (the reason might just be a typo)");
    return problem;
  }
  coco_error("Unknown observer.");
  return NULL; /* Never reached */
}

/**
 * Return the first problem in benchmark ${suite} with ${id} as problem ID,
 * or NULL.
 */
coco_problem_t *coco_suite_get_problem_by_id(const char *suite, const char *id) {
  const char *suite_options = "";
  long index = coco_suite_get_next_problem_index(suite, -1, suite_options);
  coco_problem_t *problem;
  const char *prob_id;

  for (; index >= 0; coco_suite_get_next_problem_index(suite, index, suite_options)) {
    problem = coco_suite_get_problem(suite, index);
    prob_id = coco_problem_get_id(problem);
    if (strlen(prob_id) == strlen(id) && strncmp(prob_id, id, strlen(prob_id)) == 0)
      return problem;
    coco_problem_free(problem);
  }
  return NULL;
}

#if 1
/* deprecated__coco_suite_benchmark(problem_suite, observer, options, optimizer):
 *
 * Benchmark a solver ${optimizer} with a testbed ${problem_suite}
 * using the data logger ${observer} to write data.
 *
 * Deprecated, use coco_suite_benchmark instead!
 */
void deprecated__coco_suite_benchmark(const char *problem_suite,
                                      const char *observer,
                                      const char *options,
                                      coco_optimizer_t optimizer) {
  int problem_index;
  coco_problem_t *problem;
  for (problem_index = 0;; ++problem_index) {
    problem = coco_suite_get_problem(problem_suite, problem_index);
    if (NULL == problem)
      break;
    problem = deprecated__coco_problem_add_observer(problem, observer, options); /* should remain invisible to the user*/
    optimizer(problem);
    /* Free problem after optimization. */
    coco_problem_free(problem);
  }
}

/**
 * Benchmarks a solver ${optimizer} with a testbed ${problem_suite} using the data logger ${observer_name}
 * initialized with ${observer_options} to write data.
 */
void coco_suite_benchmark(const char *suite_name,
                          const char *observer_name,
                          const char *observer_options,
                          coco_optimizer_t optimizer) {

  coco_observer_t *observer;
  coco_problem_t *problem;
  long problem_index;

  observer = coco_observer(observer_name, observer_options);

  for (problem_index = coco_suite_get_next_problem_index(suite_name, -1, "");
       problem_index >= 0;
       problem_index = coco_suite_get_next_problem_index(suite_name, problem_index, "")) {

    problem = coco_suite_get_problem(suite_name, problem_index);

    if (problem == NULL)
      break;

    problem = coco_problem_add_observer(problem, observer);
    optimizer(problem);
    coco_problem_free(problem);
  }

  coco_observer_free(observer);
}

#else
/** "improved" interface for coco_suite_benchmark: is it worth-while to have suite-options on the C-level? 
 */
void coco_suite_benchmark(const char *problem_suite, const char *problem_suite_options,
    const char *observer, const char *observer_options,
    coco_optimizer_t optimizer) {
  int problem_index;
  int is_instance;
  coco_problem_t *problem;
  char buf[222]; /* TODO: this is ugly, how to improve? The new implementation of coco_warning makes this obsolete */
  for (problem_index = -1;;) {
    problem_index = coco_suite_get_next_problem_index(problem_suite, problem_suite_options, problem_index);
    if (problem_index < 0)
      break;
    problem = coco_suite_get_problem(problem_suite, problem_index);
    if (problem == NULL) {
      snprintf(buf, 221, "problem index %d not found in problem suite %s (this is probably a bug)",
          problem_index, problem_suite);
      coco_warning(buf);
      break;
    }
    problem = coco_problem_add_observer(observer, problem, observer_options); /* should remain invisible to the user*/
    optimizer(problem);
    coco_problem_free(problem);
  }
}
#endif
