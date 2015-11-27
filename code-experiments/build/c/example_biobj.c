#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "coco.h"

/*
 * Parameters of the experiment
 */
static const long MAX_BUDGET = 1e2;
static const char *SUITE_NAME = "suite_biobj_300";
static const char *OBSERVER_NAME = "observer_biobj";
static const char *OBSERVER_OPTIONS = "result_folder: RS_on_suite_biobj_300 \
                                       include_decision_variables: 0 \
                                       log_nondominated: final";
/* static const char *SOLVER_NAME = "grid_search"; */
static const char *SOLVER_NAME = "random_search";

/*
 * Interface to the COCO evaluate function
 */
static coco_problem_t *CURRENT_COCO_PROBLEM;
typedef void (*mo_objective_function_t)(const double *, double *);
void mo_function(const double *x, double *y) {
  coco_evaluate_function(CURRENT_COCO_PROBLEM, x, y);
}

/*
 * The optimization algorithm. In this example it's a simple random search.
 */
void random_search(mo_objective_function_t f,
                   size_t number_of_objectives,
                   size_t dimension,
                   const double *lower,
                   const double *upper,
                   long budget) {

  /* Uses COCO functions for convenience */
  coco_random_state_t *rng = coco_random_new(0xdeadbeef);
  double *x = coco_allocate_vector(dimension);
  double *y = coco_allocate_vector(number_of_objectives);
  long i;

  for (i = 0; i < budget; ++i) {
    size_t j;
    for (j = 0; j < dimension; ++j) {
      x[j] = lower[j] + coco_random_uniform(rng) * (upper[j] - lower[j]);
    }
    f(x, y);
    /*
     * To be of any real use, we would need to retain the best x-value here.
     * For benchmarking purpose the implementation suffices, as the
     * observer takes care of book-keeping.
     */
  }
  coco_random_free(rng);
  coco_free_memory(x);
  coco_free_memory(y);
}

/*
 * The optimization algorithm. In this example it's a simple grid search.
 */
void grid_search(mo_objective_function_t f,
                 size_t number_of_objectives,
                 size_t dimension,
                 const double *lower,
                 const double *upper,
                 long budget) {

  /* Uses COCO functions for convenience */
  double *x = coco_allocate_vector(dimension);
  double *y = coco_allocate_vector(number_of_objectives);

  long *nodes = coco_allocate_memory(sizeof(long) * dimension);
  double *grid_step = coco_allocate_vector(dimension);
  size_t i, j;

  const long max_nodes = (long) floor(pow((double) budget, 1.0 / (double) dimension)) - 1;

  /* Initialization */
  for (j = 0; j < dimension; j++) {
    nodes[j] = 0;
    grid_step[j] = (upper[j] - lower[j]) / (double) max_nodes;
  }

  while (1) {
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

    /* Construct x and evaluate it */
    for (j = 0; j < dimension; j++) {
      x[j] = lower[j] + grid_step[j] * (double) nodes[j];
    }
    f(x, y);
    /*
     * To be of any real use, we would need to retain the best x-value here.
     * For benchmarking purpose the implementation suffices, as the
     * observer takes care of book-keeping.
     */

    /* Handle the special case with only one point in the grid */
    if (max_nodes == 0)
      break;
  }

  coco_free_memory(x);
  coco_free_memory(y);
  coco_free_memory(nodes);
  coco_free_memory(grid_step);
}

/*
 * Prepares and executes optimization with restarts using the chosen optimization algorithm (the one
 * named SOLVER_NAME).
 */
void my_optimizer(coco_problem_t *problem) {

  /* Prepare, set up convenience definitions */
  size_t dimension = coco_problem_get_dimension(problem);
  const double *lbounds = coco_problem_get_smallest_values_of_interest(problem);
  const double *ubounds = coco_problem_get_largest_values_of_interest(problem);
  double * initial_x = coco_allocate_vector(coco_problem_get_dimension(problem));
  long remaining_budget;

  coco_problem_get_initial_solution(problem, initial_x);
  CURRENT_COCO_PROBLEM = problem; /* Do not change this, it's used in objective_function */

  while ((remaining_budget = MAX_BUDGET - coco_problem_get_evaluations(problem)) > 0) {
    /* Call the optimization algorithm */

    if (strcmp("random_search", SOLVER_NAME) == 0) {
      random_search(mo_function, coco_problem_get_number_of_objectives(problem), dimension, lbounds, ubounds,
          remaining_budget);
    }
    else if (strcmp("grid_search", SOLVER_NAME) == 0) {
      grid_search(mo_function, coco_problem_get_number_of_objectives(problem), dimension, lbounds, ubounds,
          remaining_budget);
      /* Grid search should not be restarted*/
      break;
    }
  }

  coco_free_memory(initial_x);
}

int main(void) {

  printf("Running the experiments... (it takes time, be patient)");
  coco_suite_benchmark(SUITE_NAME, OBSERVER_NAME, OBSERVER_OPTIONS, my_optimizer);
  printf("Done!");
  return 0;
}
