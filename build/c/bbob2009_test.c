/*
 * Test function and instance ID extraction for BBOB2009 problem suit.
 */

#include <stdio.h>
#include <float.h>
#include <math.h>

#include "coco.h"

int main(int argc, char **argv) {
  int problem_idx = -1;
  const char *suite = "bbob2009";
  coco_problem_t *problem = NULL;

  while ((problem_idx = coco_next_problem_index(suite, problem_idx, "")) >= 0) {
    problem = coco_get_problem(suite, problem_idx);
    if (problem != NULL) {
      printf("%4i: %s\n", problem_idx, coco_get_problem_id(problem));
      coco_free_problem(problem);
    } else
      printf("problem %4i not found\n", problem_idx);
  }
  return 0;
}
