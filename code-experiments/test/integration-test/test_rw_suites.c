/**
 * Tests that the real-world suites evaluate the initial solution.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "coco.h"
#include "coco.c"

/* Tests whether any problems occur in the evaluation of the initial solution.
 */
void run_once(char *suite_name, char *suite_options) {

  coco_suite_t *suite;
  coco_problem_t *problem;
  double *initial_solution;
  double *y = coco_allocate_vector(1);

  suite = coco_suite(suite_name, NULL, suite_options);

  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {
    initial_solution = coco_allocate_vector(coco_problem_get_dimension(problem));
    coco_problem_get_initial_solution(problem, initial_solution);
    coco_evaluate_function(problem, initial_solution, y);
    coco_free_memory(initial_solution);
  }

  coco_suite_free(suite);
  coco_free_memory(y);

  printf("Performed integration test on the %s suite\n", suite_name);
  printf("DONE!\n");
  fflush(stdout);
}

int main(void)  {

  run_once("rw-gan-mario", "");

  run_once("rw-gan-mario-biobj", "");

  run_once("rw-top-trumps", "");

  run_once("rw-top-trumps-biobj", "");

  coco_remove_directory("exdata");
  return 0;
}
