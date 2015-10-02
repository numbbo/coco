#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "coco.h"


static coco_problem_t *CURRENT_COCO_PROBLEM; /* used in objective_function */

/**************************************************
 *   Set up the experiment                    
 **************************************************/
static const long MAX_BUDGET = 1e2;  /* work on small budgets first */
static const char * SUITE_NAME       = "biobjective_combinations";
/* static const char * SUITE_OPTIONS    = "";*/ /* e.g.: "instances:1-5; dimensions:-20" */
/* static const char * OBSERVER_NAME = "no_observer"; / * writes no data */
static const char * OBSERVER_NAME    = "mo_toy_observer"; /* writes data */
static const char * OBSERVER_OPTIONS = "mo_random_search_on_biobjective_suite_300";
static const char * SOLVER_NAME      = "mo_random_search"; /* for the choice in coco_optimize below */
/* static const int NUMBER_OF_BATCHES   = 88;*/  /* use 1 for single batch :-) batches can be run independently in parallel */
/* static int CURRENT_BATCH             = 1;*/  /* runs from 1 to NUMBER_OF_BATCHES, or any other consecutive sequence */


/**************************************************
 * Example objective function interface and solver,
 * for the MULTIOBJECTIVE case
 **************************************************/
typedef void (*mo_objective_function_t) (const double *, double *);
void mo_objective_function(const double *x, double *y) {
    coco_evaluate_function(CURRENT_COCO_PROBLEM, x, y);
}
void mo_random_search(mo_objective_function_t func,
                      size_t number_of_objectives,
                      size_t dimension,
                      const double *lower,
                      const double *upper,
                      long budget) {
  coco_random_state_t *rng = coco_new_random(0xdeadbeef); /* use coco fcts for convenience */
  double *x = coco_allocate_vector(dimension);
  double *y = coco_allocate_vector(number_of_objectives);
  long i;

  for (i = 0; i < budget; ++i) {
    size_t j;
    for (j = 0; j < dimension; ++j) {
      x[j] = lower[j] + coco_uniform_random(rng) * (upper[j] - lower[j]);
    }
    func(x, y);
     /* To be of any real use, we would need to retain the best x-value here.
      * For benchmarking purpose the implementation suffices, as the
      * observer takes care of book-keeping. 
      */
  }
  coco_free_random(rng);
  coco_free_memory(x);
  coco_free_memory(y);
}


/**
 * Finally, coco_optimize calls, depending on SOLVER_NAME, one
 * of the defined optimizers (e.g. random_search, my_solver, ...),
 * using the respective matching objective function. 
 */
void coco_optimize(coco_problem_t *problem) { /* should at the least take budget as argument, but this is not coco_benchmark compliant */
  /* prepare, set up convenience definitions */
  size_t dimension = coco_get_number_of_variables(problem);
  const double * lbounds = coco_get_smallest_values_of_interest(problem);
  const double * ubounds = coco_get_largest_values_of_interest(problem);
  double * initial_x = coco_allocate_vector(coco_get_number_of_variables(problem));
  /* const double final_target = coco_get_final_target_fvalue1(problem);*/
  /* const double final_target = 1e-3; */
  long remaining_budget; 
  
  coco_get_initial_solution(problem, initial_x);
  CURRENT_COCO_PROBLEM = problem; /* do not change this, it's used in objective_function */

  while ((remaining_budget = MAX_BUDGET - coco_get_evaluations(problem)) > 0) {
    /* call the solver */
    if (strcmp("mo_random_search", SOLVER_NAME) == 0) { /* random search for bi-objective case */
      mo_random_search(mo_objective_function,
                       coco_get_number_of_objectives(problem), dimension,
                       lbounds, ubounds, remaining_budget);
    }
  }
  coco_free_memory(initial_x);
}


/**************************************************
 *   run the experiment                       
 **************************************************/
/**
 * Bi-objective BBOB 2009 functions
 */

/* Added here for compatibility!!!*/
#define BIOBJECTIVE_NUMBER_OF_COMBINATIONS 300
#define BIOBJECTIVE_NUMBER_OF_INSTANCES 5
#define BIOBJECTIVE_NUMBER_OF_DIMENSIONS 5
static long biobjective_encode_problem_index(int combination_idx, long instance_idx, int dimension_idx) {
    long problem_index;
    problem_index = instance_idx + 
                    combination_idx * BIOBJECTIVE_NUMBER_OF_INSTANCES + 
                    dimension_idx * (BIOBJECTIVE_NUMBER_OF_INSTANCES * BIOBJECTIVE_NUMBER_OF_COMBINATIONS);
    return problem_index;
}

/* Interface via dimension, function-ID and instance-ID. This does not translate
   directly to different languages or benchmark suites. */
int main() {
  int problem_index, combination_idx, instance_idx, dimension_idx;
  coco_problem_t *problem;
  
  for (dimension_idx = 0; dimension_idx < 3; dimension_idx++) {
      for (combination_idx = 0; combination_idx < 300; combination_idx++) {
          for (instance_idx = 0; instance_idx < 5; instance_idx++) {
              problem_index = biobjective_encode_problem_index(combination_idx,
                                                               instance_idx,
                                                               dimension_idx);
printf("problem_index = %d, combination_idx = %d, instance_idx = %d, dimension_idx = %d\n", problem_index, combination_idx, instance_idx, dimension_idx);
              problem = coco_get_problem(SUITE_NAME, problem_index);
              
              problem = coco_observe_problem(OBSERVER_NAME, problem, OBSERVER_OPTIONS);
              if (problem == NULL)
                break;
              coco_optimize(problem);
              coco_free_problem(problem);
          }
      }
  }
  return 0;
}
