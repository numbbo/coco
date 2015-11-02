#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "coco.h"
#include "coco_strdup.c"
#include "coco_problem.c"
#include "suite_bbob2009.c"

/**
 * suite_mo_first_attempt(problem_index):
 *
 * Return the ${problem_index}-th benchmark problem from...
 * If the function index is out of bounds, return
 * NULL.
 */
static coco_problem_t *suite_mo_first_attempt(const long problem_index) {
  coco_problem_t *problem, *problem2;
  long dimension, instance, instance2;
  int f, f2;

  if (problem_index < 0)
    return NULL;

  if (problem_index < 24) {

    /* here we compute the mapping from problem index to the following five values */

    dimension = 10;
    f = 1;
    f2 = 1 + (int) (problem_index % 24);
    instance = 0;
    instance2 = 1;

    problem = suite_bbob2009_problem(f, dimension, instance);

    problem2 = suite_bbob2009_problem(f2, dimension, instance2);
    problem = coco_stacked_problem_allocate(problem, problem2);
    /* repeat the last two lines to add more objectives */
#if 0
    coco_suite_problem_setf_id(problem, "ID-F%03d-F%03d-d03%ld-%06ld", f, f2, dimension, problem_index);
    coco_suite_problem_setf_name(problem, "%s + %s",
        coco_problem_get_name(problem), coco_problem_get_name(problem2));
#endif
    problem->index = problem_index;

    return problem;
  } /* else if ... */
  return NULL;
}

