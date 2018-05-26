/**
 * Tests that the rw-gan suite evaluates the initial solution.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "coco.h"
#include "coco.c"

/* Tests whether any problems occur in the evaluation of the initial solution of the rw-gan
 * benchmark problems.
 */
void run_once(char *suite_options) {

  coco_suite_t *suite;
  coco_problem_t *problem;
  double *initial_solution;
  double *y = coco_allocate_vector(1);

  suite = coco_suite("rw-gan", NULL, suite_options);

  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {
    initial_solution = coco_allocate_vector(coco_problem_get_dimension(problem));
    coco_problem_get_initial_solution(problem, initial_solution);
    coco_evaluate_function(problem, initial_solution, y);
    coco_free_memory(initial_solution);
  }

  coco_suite_free(suite);
  coco_free_memory(y);

  printf("Performed integration test on the rw-gan suite\n");
  printf("DONE!\n");
  fflush(stdout);
}

int main(void)  {

  run_once("");

  coco_remove_directory("exdata");
  return 0;
}
