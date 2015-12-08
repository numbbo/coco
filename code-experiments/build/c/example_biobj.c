/*
 * An example of benchmarking the bi-objective COCO suite. Two algorithms are implemented: random search
 * and grid search. Only random search is run by default.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "coco.h"

/*
 * Parameters of the experiment
 */
static const size_t MAX_BUDGET = 1e2;
static const char *SUITE_NAME = "suite_biobj_300";
static const char *OBSERVER_NAME = "observer_biobj";

/**
 * A random search optimizer
 */
void my_random_search(coco_problem_t *problem) {

  coco_random_state_t *rng = coco_random_new(0xdeadbeef);
  const double *lbounds = coco_problem_get_smallest_values_of_interest(problem);
  const double *ubounds = coco_problem_get_largest_values_of_interest(problem);
  size_t dimension = coco_problem_get_dimension(problem);
  size_t number_of_objectives = coco_problem_get_number_of_objectives(problem);
  double *x = coco_allocate_vector(dimension);
  double *y = coco_allocate_vector(number_of_objectives);
  double range;
  size_t i, j;

  for (i = 0; i < MAX_BUDGET; ++i) {

    /* Construct x as a random point between the lower and upper bounds */
    for (j = 0; j < dimension; ++j) {
      range = ubounds[j] - lbounds[j];
      x[j] = lbounds[j] + coco_random_uniform(rng) * range;
    }

    /* Call COCO's evaluate function where all the logging is performed */
    coco_evaluate_function(problem, x, y);

  }

  coco_random_free(rng);
  coco_free_memory(x);
  coco_free_memory(y);
}

/**
 * A grid search optimizer. If MAX_BUDGET is not enough to cover even the smallest possible grid, only the
 * first MAX_BUDGET nodes of the grid are evaluated.
 */
void my_grid_search(coco_problem_t *problem) {

  const double *lbounds = coco_problem_get_smallest_values_of_interest(problem);
  const double *ubounds = coco_problem_get_largest_values_of_interest(problem);
  size_t dimension = coco_problem_get_dimension(problem);
  size_t number_of_objectives = coco_problem_get_number_of_objectives(problem);
  double *x = coco_allocate_vector(dimension);
  double *y = coco_allocate_vector(number_of_objectives);
  long *nodes = coco_allocate_memory(sizeof(long) * dimension);
  double *grid_step = coco_allocate_vector(dimension);
  size_t i, j;
  size_t evaluations = 0;

  long max_nodes = (long) floor(pow((double) MAX_BUDGET, 1.0 / (double) dimension)) - 1;

  /* Take care of the borderline case */
  if (max_nodes < 1) max_nodes = 1;

  /* Initialization */
  for (j = 0; j < dimension; j++) {
    nodes[j] = 0;
    grid_step[j] = (ubounds[j] - lbounds[j]) / (double) max_nodes;
  }

  while (evaluations < MAX_BUDGET) {

    /* Construct x and evaluate it */
    for (j = 0; j < dimension; j++) {
      x[j] = lbounds[j] + grid_step[j] * (double) nodes[j];
    }

    /* Call COCO's evaluate function where all the logging is performed */
    coco_evaluate_function(problem, x, y);
    evaluations++;

    /* Inside the grid, move to the next node */
    if (nodes[0] < max_nodes) {
      nodes[0]++;
    }

    /* At an outside node of the grid, move to the next level */
    else if (max_nodes > 0) {
      for (j = 1; j < dimension; j++) {
        if (nodes[j] < max_nodes) {
          nodes[j]++;
          for (i = 0; i < j; i++)
            nodes[i] = 0;
          break;
        }
      }

      /* At the end of the grid, exit */
      if ((j == dimension) && (nodes[j - 1] == max_nodes))
        break;
    }
  }

  coco_free_memory(x);
  coco_free_memory(y);
  coco_free_memory(nodes);
  coco_free_memory(grid_step);
}


int main(void) {

  /*static const char *observer_options_GS = "result_folder: GS_on_suite_biobj_300 \
                                            algorithm_name: GS \
                                            algorithm_info: \"A simple grid search algorithm\" \
                                            include_decision_variables: 1 \
                                            compute_indicators: 1 \
                                            log_nondominated: final";*/

  static const char *observer_options_RS = "result_folder: RS_on_suite_biobj_300 \
                                            algorithm_name: RS \
                                            algorithm_info: \"A simple random search algorithm\" \
                                            include_decision_variables: 1 \
                                            compute_indicators: 1 \
                                            log_nondominated: final";

  printf("Running the experiments... (it takes time, be patient)\n");
  fflush(stdout);

  coco_suite_benchmark(SUITE_NAME, OBSERVER_NAME, observer_options_RS, my_random_search);

  /* coco_suite_benchmark(SUITE_NAME, OBSERVER_NAME, observer_options_GS, my_grid_search); */

  printf("Done!\n");
  fflush(stdout);

  return 0;
}
