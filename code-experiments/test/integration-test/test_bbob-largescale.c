/**
 * Tests different combinations of suite options for the largescale suite
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "coco.h"

static void wait_in_seconds(time_t secs) {
    time_t retTime = time(0) + secs;
    while (time(0) < retTime);
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
  double *function_values = coco_allocate_vector(number_of_objectives);
  double range;
  size_t i, j;

  for (i = 0; i < budget; ++i) {

    for (j = 0; j < dimension; ++j) {
      range = ubounds[j] - lbounds[j];
      x[j] = lbounds[j] + coco_random_uniform(rng) * range;
    }

    coco_evaluate_function(problem, x, function_values);

  }

  coco_random_free(rng);
  coco_free_memory(x);
  coco_free_memory(function_values);
}

/* Each time: run the benchmark and delete the output folder */
void run_once(char *suite_options) {

  coco_suite_t *suite;
  coco_observer_t *observer;
  coco_problem_t *problem;

  /* Set some options for the observer. See documentation for other options. */
  char *observer_options =
      coco_strdupf("result_folder: RS_on_%s "
                   "algorithm_name: RS "
                   "algorithm_info: \"A simple random search algorithm\"",
                   "bbob-largescale");

  printf("Running experiment with suite options %s\n", suite_options);
  fflush(stdout);

  suite = coco_suite("bbob-largescale", NULL, suite_options);
  observer = coco_observer("bbob", observer_options);
  coco_free_memory(observer_options);

  while ((problem = coco_suite_get_next_problem(suite, observer)) != NULL) {
    my_optimizer(problem);
  }

  coco_observer_free(observer);
  coco_suite_free(suite);

  wait_in_seconds(2); /* So that the directory removal is surely finished */

  printf("DONE!\n");
  fflush(stdout);
}

int main(void)  {

  /* Mute output that is not error */
  coco_set_log_level("error");

  run_once("dimensions: 20,40 function_indices: 1-8 instance_indices: 1");
  run_once("dimensions: 80 function_indices: 9-20 instance_indices: 15");
  run_once("dimensions: 160,320,640 function_indices: 21-24 instance_indices: 10");

  coco_remove_directory("exdata");
  return 0;
}
