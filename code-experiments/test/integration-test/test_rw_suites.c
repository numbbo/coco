/**
 * Tests concerning real-world suites and observers.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "coco.h"
#include "coco.c"

/**
 * Tests whether any problems occur in the evaluation of the initial solution.
 */
void run_once(char *suite_name, char *suite_options) {

  coco_suite_t *suite;
  coco_problem_t *problem;
  double *initial_solution;
  double *y = coco_allocate_vector(2);

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

/**
 * A random search optimizer.
 */
void my_optimizer(coco_problem_t *problem) {

  const size_t budget = 2;
  coco_random_state_t *rng = coco_random_new(0xdeadbeef);
  const double *lbounds = coco_problem_get_smallest_values_of_interest(problem);
  const double *ubounds = coco_problem_get_largest_values_of_interest(problem);
  size_t dimension = coco_problem_get_dimension(problem);
  size_t number_of_objectives = coco_problem_get_number_of_objectives(problem);
  double *x = coco_allocate_vector(dimension);
  double *y = coco_allocate_vector(number_of_objectives);
  double range;
  size_t i, j;

  for (i = 0; i < budget; ++i) {

    for (j = 0; j < dimension; ++j) {
      range = ubounds[j] - lbounds[j];
      x[j] = lbounds[j] + coco_random_uniform(rng) * range;
    }

    coco_evaluate_function(problem, x, y);

  }

  coco_random_free(rng);
  coco_free_memory(x);
  coco_free_memory(y);
}

/**
 * Tests whether the rw-logger can be correctly wrapped around other loggers.
 */
void test_two_observers(char *suite_name, char *suite_options) {

  coco_suite_t *suite;
  coco_observer_t *observer_inner, *observer_outer;
  coco_problem_t *problem_inner, *problem_outer;

  suite = coco_suite(suite_name, "", suite_options);

  observer_inner = coco_observer("rw", "");
  observer_outer = coco_observer("bbob", "");

  while ((problem_inner = coco_suite_get_next_problem(suite, observer_inner)) != NULL) {

    problem_outer = coco_problem_add_observer(problem_inner, observer_outer);
    my_optimizer(problem_outer);
    problem_inner = coco_problem_remove_observer(problem_outer, observer_outer);
  }

  coco_observer_free(observer_inner);
  coco_observer_free(observer_outer);

  coco_suite_free(suite);

  printf("Performed test of two observers on the '%s' suite\n", suite_name);
  printf("DONE!\n");
  fflush(stdout);
}

int main(void)  {

  test_two_observers("rw-gan-mario", "dimensions: 10 function_indices: 1 instance_indices: 1");

  run_once("rw-top-trumps", "instance_indices: 1");
  run_once("rw-top-trumps-biobj", "instance_indices: 1");
  run_once("rw-gan-mario", "instance_indices: 1 function_indices: 2,5,8");
  /*run_once("rw-gan-mario-biobj", "instance_indices: 1 function_indices: 2,5,8"); */

  coco_remove_directory("exdata");
  return 0;
}
