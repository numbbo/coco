#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "coco_problem.c"

/**
 * mo_suit...(problem_index):
 *
 * Return the ${problem_index}-th benchmark problem from...
 * If the function index is out of bounds, return
 * NULL.
 */
static coco_problem_t *mo_suite_first_attempt(const int problem_index) {
  coco_problem_t *problem, *problem1, *problem2;

  if (problem_index < 0)
    return NULL; 

  problem1 = coco_get_problem("bbob2009", 0);
  problem2 = coco_get_problem("bbob2009", problem_index);
  problem = coco_stacked_problem_allocate("We-need-a-unique_id-here-F0102-d02",
                                          "sphere-elli-sep",
                                          problem1, problem2);
  return problem;
}

