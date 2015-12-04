/**
 * Tests different combinations of observer options for the biobj logger.
 */

#include <stdlib.h>
#include <stdio.h>

#include "coco.h"

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

  char *suite_name = "suite_biobj_300";
  char *observer_name = "observer_biobj";

  printf("Running experiment with options %s...", observer_options);
  fflush(stdout);
  coco_suite_benchmark(suite_name, observer_name, observer_options, my_optimizer);
  coco_remove_directory("biobj");
  printf("DONE!\n", observer_options);
  fflush(stdout);
}

int main(void) {
  run_once("result_folder: biobj produce_all_data 1");
  run_once("result_folder: biobj log_nondominated: none  compute_indicators: 0");
  run_once("result_folder: biobj log_nondominated: all   compute_indicators: 0");
  run_once("result_folder: biobj log_nondominated: final compute_indicators: 0");
  run_once("result_folder: biobj log_nondominated: none  compute_indicators: 1");
  run_once("result_folder: biobj log_nondominated: all   compute_indicators: 1");
  run_once("result_folder: biobj log_nondominated: final compute_indicators: 1");
  run_once("result_folder: biobj log_nondominated: none  compute_indicators: 1 include_decision_variables: 1");
  run_once("result_folder: biobj log_nondominated: all   compute_indicators: 0 include_decision_variables: 1");
  run_once("result_folder: biobj log_nondominated: final compute_indicators: 1 include_decision_variables: 1");
  return 0;
}
