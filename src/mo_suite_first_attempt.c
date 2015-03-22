#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "coco_problem.c"
#include "bbob2009_suite.c"

static coco_problem_t *mo_first_attempt_get_problem(
        long dimension, int f1, long instance1, int f2, long instance2) {
  coco_problem_t *problem, *problem1, *problem2;

  /* or get bbob problems by fun, dimension, instance */
  problem1 = bbob2009_problem(f1, dimension, instance1);
  problem2 = bbob2009_problem(f2, dimension, instance2);
  problem = coco_stacked_problem_allocate("We-need-a-unique_id-here-F010x-d10", /* coco_strdup should be improved with varargs to achieve this */
                                          "sphere-sphere",
                                          problem1, problem2);
  return problem;
}

/**
 * mo_suit...(problem_index):
 *
 * Return the ${problem_index}-th benchmark problem from...
 * If the function index is out of bounds, return
 * NULL.
 */
static coco_problem_t *mo_suite_first_attempt(const int problem_index) {
  long dimension = 10;
  int f1, f2;
  
  if (problem_index < 0) 
    return NULL;
  if (problem_index > 24) 
    return NULL;
  
  /* here we compute the mapping from problem index to the five values dim, f1/2, I1/2 */
  
  dimension = 10;
  f1 = 1;
  f2 = 1 + (problem_index % 24); 
  return mo_first_attempt_get_problem(dimension, f1, 0, f2, 1);
}

