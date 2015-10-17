#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "coco.h"

static coco_problem_t *CURRENT_COCO_PROBLEM; /* used in objective_function */

/**************************************************
 * Example objective function interface and solver,
 * no need to modify 
 **************************************************/
typedef double (*objective_function_example_t) (const double *);
double objective_function_example(const double *x) {
    double y;
    coco_evaluate_function(CURRENT_COCO_PROBLEM, x, &y);
    return y;
}

typedef double (*constraint_function_example_t) (const double *);
double constraint_function_example(const double *x) {
    double y;
    coco_evaluate_constraint(CURRENT_COCO_PROBLEM, x, &y);
    return y;
}
void random_search(size_t dimension,
                   objective_function_example_t fun,
                   constraint_function_example_t cons,
                   const double *lower,
                   const double *upper,
                   long budget,
                   double final_target,
		   double feasibility_threshold) {
  coco_random_state_t *rng = coco_new_random(0xdeadbeef); /* use coco fcts for convenience */
  double *x = coco_allocate_vector(dimension);
  double objective_function_value;
  double constraint_function_value;
  long i;

  for (i = 0; i < budget; ++i) {
    size_t j;
    for (j = 0; j < dimension; ++j) {
      x[j] = lower[j] + coco_uniform_random(rng) * (upper[j] - lower[j]);
    }
    objective_function_value = fun(x);
    constraint_function_value = cons(x);
    
    printf("\n\nIteration i = %d\n\n", i);
    for (j = 0; j < dimension; ++j) {
      printf("x[%d] = %f\n", j, x[j]);
    
    }
    printf("f(x) = %f\n", objective_function_value);
    printf("c(x) = %f\n", constraint_function_value);
    
     /* To be of any real use, we would need to retain the best x-value here.
      * For benchmarking purpose the implementation suffices, as the
      * observer takes care of bookkeeping. 
      */
    if (objective_function_value <= final_target && constraint_function_value <= feasibility_threshold)
      break;
  }
  coco_free_random(rng);
  coco_free_memory(x);
}

/**************************************************
 *   Set up the experiment                    
 **************************************************/
static const long MAX_BUDGET = 1e2;  /* work on small budgets first */
static const char * SUITE_NAME       = "consprob2015";
static const char * SUITE_OPTIONS    = ""; /* e.g.: "instances:1-5; dimensions:-20" */
static const char * OBSERVER_NAME    = "consprob2015_observer"; /* writes data */
/* static const char * OBSERVER_NAME = "no_observer"; / * writes no data */
static const char * OBSERVER_OPTIONS = "random_search_on_bbob2009"; /* future: "folder:random_search; verbosity:1" */
static const char * SOLVER_NAME      = "random_search"; /* for the choice in coco_optimize below */
/*  static const char * SOLVER_NAME   = "my_solver"; / * for the choice in coco_optimize below */
static const int NUMBER_OF_BATCHES   = 88;  /* use 1 for single batch :-) batches can be run independently in parallel */
static int CURRENT_BATCH             = 1;  /* runs from 1 to NUMBER_OF_BATCHES, or any other consecutive sequence */

/**************************************************
 *  Objective function interface to solver,
 *  needs to be adapted       
 **************************************************/
/**
 * Definition of the objective function, which must
 * probably be adapted to the solver used. 
 */
/* MODIFY this to work with the solver to be benchmarked */
typedef void (*objective_function_t) (const double *, double *);
void objective_function(const double *x, double *y) {
  /* MODIFY/REWRITE this function to work with the typedef two lines above */
  coco_evaluate_function(CURRENT_COCO_PROBLEM, x, y); /* this call writes objective_function(x) in y */
}
/* Minimal solver definition only to avoid compile/link errors/warnings below for my_solver */
void my_solver(objective_function_t func, const double *initial_x, size_t dim, long budget) {
  double y; func(initial_x, &y); 
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
  const double final_target = coco_get_final_target_fvalue1(problem);
  const double feasibility_threshold = 1e-8;
  long remaining_budget; 
  
  coco_get_initial_solution(problem, initial_x);
  CURRENT_COCO_PROBLEM = problem; /* do not change this, it's used in objective_function */

  while ((remaining_budget = MAX_BUDGET - coco_get_evaluations(problem)) > 0) {
    /* call the solver */
    if (strcmp("random_search", SOLVER_NAME) == 0) { /* example case, no need to modify */
      random_search(dimension, objective_function_example, constraint_function_example,
                    lbounds, ubounds, remaining_budget, final_target, feasibility_threshold);
    } else if (strcmp("my_solver", SOLVER_NAME) == 0) {
      /* MODIFY this call according to objective_function_t and solver's needs */
      my_solver(objective_function, /* MODIFY my_solver to your_solver ... */
                initial_x, dimension, remaining_budget); 
    }
  }
  coco_free_memory(initial_x);
}

/**************************************************
 *   run the experiment                       
 **************************************************/
int main() {
  
  coco_problem_t *problem;
  int problem_index;
    
  for (problem_index = -1; ; ) {
    problem_index = coco_next_problem_index(SUITE_NAME, problem_index, SUITE_OPTIONS); 
    if (problem_index < 0)
      break;
    
    problem = coco_get_problem(SUITE_NAME, problem_index);
    problem = coco_observe_problem(OBSERVER_NAME, problem, OBSERVER_OPTIONS);
    coco_optimize(problem);
    coco_free_problem(problem);
  }
  printf("Done with suite '%s' (options '%s')", SUITE_NAME, SUITE_OPTIONS);
  if (NUMBER_OF_BATCHES > 1) printf(" batch %d/%d.\n", CURRENT_BATCH, NUMBER_OF_BATCHES);
  else printf(".\n");
  return 0;
  
  /*
  
  size_t i;
  coco_problem_t *problem = NULL;
  int problem_index; // Is this useful when we are not testing with a benchmark suite?
  const size_t dimension = 3;
  double xopt[dimension], fopt;
  
  fopt = 10;
  for (i = 0; i < dimension; i++) { xopt[i] = 5.0; }

  problem = sphere_problem(dimension);
  problem = oscillate_variables(problem);
  problem = shift_variables(problem, xopt, 0);
  problem = shift_objective(problem, fopt);
  problem = coco_observe_problem(OBSERVER_NAME, problem, OBSERVER_OPTIONS);
  coco_optimize(problem);
  coco_free_problem(problem);
  
  return 0;
  
  */
}

