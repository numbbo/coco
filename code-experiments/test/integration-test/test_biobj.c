/**
 * Tests different combinations of observer options for the biobj logger.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "coco.h"

static void wait_in_seconds(unsigned int secs) {
    time_t retTime = time(0) + secs;
    while (time(0) < retTime);
}

/**
 * A random search optimizer.
 */
void my_optimizer(coco_problem_t *problem) {

  const size_t budget = 10;
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

/* Each time: run the benchmark and delete the output folder */
void run_once(char *observer_options) {

  coco_suite_t *suite;
  coco_observer_t *observer;
  coco_problem_t *problem;

  printf("Running experiment with options %s ...", observer_options);
  fflush(stdout);

  suite = coco_suite("bbob-biobj", NULL, "dimensions: 2 functions: 5-10 instances: 2-3");
  observer = coco_observer("bbob-biobj", observer_options);

  while ((problem = coco_suite_get_next_problem(suite, observer)) != NULL) {

    my_optimizer(problem);

  }

  coco_observer_free(observer);
  coco_suite_free(suite);

  coco_remove_directory("exdata");
  wait_in_seconds(2); /* So that the directory removal is surely finished */

  printf("DONE!\n");
  fflush(stdout);
}

int main( int argc, char *argv[] )  {

  if ((argc == 2) && (strcmp(argv[1], "leak_check") == 0)) {
    run_once("produce_all_data 1");
  }
  else {
    run_once("produce_all_data 1");
    run_once("log_nondominated: none  compute_indicators: 0");
    run_once("log_nondominated: all   compute_indicators: 0");
    run_once("log_nondominated: final compute_indicators: 0");
    run_once("log_nondominated: none  compute_indicators: 1");
    run_once("log_nondominated: all   compute_indicators: 1");
    run_once("log_nondominated: final compute_indicators: 1");
    run_once("log_nondominated: none  compute_indicators: 1 log_decision_variables: all");
    run_once("log_nondominated: all   compute_indicators: 0 log_decision_variables: none");
    run_once("log_nondominated: final compute_indicators: 1 log_decision_variables: low_dim");
  }
  return 0;
}
