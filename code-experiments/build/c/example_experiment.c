/*
 * An example of benchmarking random search on three COCO suites. A grid search optimizer is also
 * implemented and can be used instead of random search.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "coco.h"

/*
 * Parameters of the experiment
 */
static const size_t MAX_BUDGET = 1e2;

/**
 * A random search algorithm that can be used for single- as well as multi-objective optimization.
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
 * A grid search optimizer that can be used for single- as well as multi-objective optimization.
 * If MAX_BUDGET is not enough to cover even the smallest possible grid, only the first MAX_BUDGET
 * nodes of the grid are evaluated.
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

/**
 * A simple example of benchmarking an optimization algorithm on the bbob suite with instances from 2009.
 */
void example_bbob() {

  /* Some options of the bbob observer. See documentation for other options. */
  const char *observer_options = "result_folder: RS_on_bbob \
                                  algorithm_name: RS \
                                  algorithm_info: \"A simple random search algorithm\"";

  coco_suite_t *suite;
  coco_observer_t *observer;
  coco_problem_t *problem;

  suite = coco_suite("bbob", "year: 2009", "dimensions: 2,3,5,10,20 instance_idx: 1,2");
  observer = coco_observer("bbob", observer_options);

  while ((problem = coco_suite_get_next_problem(suite, observer)) != NULL) {
    my_random_search(problem);
  }

  coco_observer_free(observer);
  coco_suite_free(suite);

}

/**
 * A simple example of benchmarking an optimization algorithm on the biobjective suite.
 */
void example_biobj() {

  /* Some options of the biobjective observer. See documentation for other options. */
  const char *observer_options = "result_folder: RS_on_bbob-biobj \
                                  algorithm_name: RS \
                                  algorithm_info: \"A simple random search algorithm\" \
                                  compute_indicators: 1 \
                                  log_nondominated: final";

  coco_suite_t *suite;
  coco_observer_t *observer;
  coco_problem_t *problem;

  suite = coco_suite("bbob-biobj", NULL, "dimensions: 2,3,5,10,20 instance_idx: 1,2");
  observer = coco_observer("bbob-biobj", observer_options);

  while ((problem = coco_suite_get_next_problem(suite, observer)) != NULL) {
    my_random_search(problem);
  }

  coco_observer_free(observer);
  coco_suite_free(suite);

}

/**
 * A simple example of benchmarking an optimization algorithm on the toy suite.
 */
void example_toy() {

  /* Some options of the toy observer. See documentation for other options. */
  const char *observer_options = "result_folder: RS_on_toy \
                                  algorithm_name: RS \
                                  algorithm_info: \"A simple random search algorithm\"";

  coco_suite_t *suite;
  coco_observer_t *observer;
  coco_problem_t *problem;

  suite = coco_suite("toy", NULL, NULL);
  observer = coco_observer("toy", observer_options);

  while ((problem = coco_suite_get_next_problem(suite, observer)) != NULL) {
    my_random_search(problem);
  }

  coco_observer_free(observer);
  coco_suite_free(suite);

}

/**
 * The main method calls all three
 */
int main(void) {

  printf("Running the experiments... (it takes time, be patient)\n");
  fflush(stdout);

  example_bbob();

  printf("First example done!\n");
  fflush(stdout);

  example_biobj();

  printf("Second example done!\n");
  fflush(stdout);

  example_toy();

  printf("Third example done!\n");
  fflush(stdout);

  return 0;
}
