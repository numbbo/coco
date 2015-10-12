/*
 * Test function and instance ID extraction for BBOB2009 problem suite.
 */

#include <stdio.h>
#include <float.h>
#include <math.h>

#include "coco.h"

int main(void) {
  long problem_idx = -1;
  const char *suite = "suite_bbob2009";
  coco_problem_t *problem = NULL;

  while ((problem_idx = coco_suite_get_next_problem_index(suite, problem_idx, "")) >= 0) {
    problem = coco_suite_get_problem(suite, problem_idx);
    if (problem != NULL) {
      printf("%4ld: %s\n", problem_idx, coco_problem_get_id(problem));
      coco_problem_free(problem);
    } else
      printf("problem %4ld not found\n", problem_idx);
  }
  return 0;
}
