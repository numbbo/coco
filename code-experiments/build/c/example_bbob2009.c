#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "coco.h"

/* Timing of random_search on a MacBook Pro (2014) [s]: 
    evals=       [ 1e2, 1e3, 1e4, 2e4, 4e4  ],  # on about 2e3 problems
    verbose=     [  2,  10,   80, 152, 299, ],
    quiet=       [  2,   9,   78,           ],
    no_observer= [  1,   7,   73, 146, 291  ],
    old_code =   [ -1, -1,    65, ]
    
    ==> 7.5e-3s/eval ~ 1e-2s / eval ~ 10s / 1e3evals ~ 2h / 1e6evals ~ 90d / 1e9evals
 */ 
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
void random_search(size_t dimension,
                   objective_function_example_t fun,
                   const double *lower,
                   const double *upper,
                   long budget,
                   double final_target) {
  coco_random_state_t *rng = coco_random_new(0xdeadbeef); /* use coco fcts for convenience */
  double *x = coco_allocate_vector(dimension);
  double y;
  long i;

  for (i = 0; i < budget; ++i) {
    size_t j;
    for (j = 0; j < dimension; ++j) {
      x[j] = lower[j] + coco_random_uniform(rng) * (upper[j] - lower[j]);
    }
    y = fun(x);
     /* To be of any real use, we would need to retain the best x-value here.
      * For benchmarking purpose the implementation suffices, as the
      * observer takes care of bookkeeping. 
      */
    if (y <= final_target)
      break;
  }
  coco_random_free(rng);
  coco_free_memory(x);
}

/**************************************************
 *   Set up the experiment                    
 **************************************************/
static const long MAX_BUDGET = 1e2;  /* work on small budgets first */
static const char *SUITE_NAME       = "suite_bbob2009";
static const char *SUITE_OPTIONS    = ""; /* e.g.: "instances:1-5; dimensions:-20" */
static const char *OBSERVER_NAME    = "observer_bbob2009"; /* writes data */
/* static const char *OBSERVER_NAME = "no_observer"; / * writes no data */
static const char *OBSERVER_OPTIONS = "RS_on_suite_bbob2009"; /* future: "folder:random_search; verbosity:1" */
static const char *SOLVER_NAME      = "random_search"; /* for the choice in coco_optimize below */
/*  static const char *SOLVER_NAME   = "my_solver"; / * for the choice in coco_optimize below */
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
  double y;
  (void)dim; (void)budget;
  func(initial_x, &y);
}
/**
 * Finally, coco_optimize calls, depending on SOLVER_NAME, one
 * of the defined optimizers (e.g. random_search, my_solver, ...),
 * using the respective matching objective function. 
 */
void coco_optimize(coco_problem_t *problem) { /* should at the least take budget as argument, but this is not coco_benchmark compliant */
  /* prepare, set up convenience definitions */
  size_t dimension = coco_problem_get_dimension(problem);
  const double * lbounds = coco_problem_get_smallest_values_of_interest(problem);
  const double * ubounds = coco_problem_get_largest_values_of_interest(problem);
  double * initial_x = coco_allocate_vector(coco_problem_get_dimension(problem));
  const double final_target = coco_problem_get_final_target_fvalue1(problem);
  long remaining_budget; 
  
  coco_problem_get_initial_solution(problem, initial_x);
  CURRENT_COCO_PROBLEM = problem; /* do not change this, it's used in objective_function */

  while ((remaining_budget = MAX_BUDGET - coco_problem_get_evaluations(problem)) > 0) {
    /* call the solver */
    if (strcmp("random_search", SOLVER_NAME) == 0) { /* example case, no need to modify */
      random_search(dimension, objective_function_example,
                    lbounds, ubounds, remaining_budget, final_target);
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
#if 0
int main() {
  coco_benchmark(SUITE_NAME, OBSERVER_NAME, OBSERVER_OPTIONS,
                 coco_optimize);
  return 0;
}
#elif 1
int main(void) {  /* short example, also nice to read */
  coco_problem_t *problem;
  long problem_index;
  
  for (problem_index = 0; problem_index >= 0;
       problem_index = coco_suite_get_next_problem_index(SUITE_NAME, problem_index, SUITE_OPTIONS)) {
    problem = coco_suite_get_problem(SUITE_NAME, problem_index);
    problem = deprecated__coco_problem_add_observer(problem, OBSERVER_NAME, OBSERVER_OPTIONS);
    coco_optimize(problem);
    coco_problem_free(problem);
  }
  printf("Done with suite '%s' (options '%s')", SUITE_NAME, SUITE_OPTIONS);
  if (NUMBER_OF_BATCHES > 1) printf(" batch %d/%d.\n", CURRENT_BATCH, NUMBER_OF_BATCHES);
  else printf(".\n");
  return 0;
}

#elif 0
int main(void) { /* longer example supporting several batches */
  coco_problem_t * problem;
  long problem_index = coco_suite_get_next_problem_index(
                        SUITE_NAME, -1, SUITE_OPTIONS); /* next(-1) == first */
  if (NUMBER_OF_BATCHES > 1)
    printf("Running only batch %d out of %d batches for suite %s\n",
           CURRENT_BATCH, NUMBER_OF_BATCHES, SUITE_NAME);
  for ( ; problem_index >= 0;
       problem_index = coco_suite_get_next_problem_index(SUITE_NAME, problem_index, SUITE_OPTIONS)
      ) {
    /* here we reject indices from other batches */
    if (((problem_index - CURRENT_BATCH + 1) % NUMBER_OF_BATCHES) != 0)
      continue;
    problem = coco_suite_get_problem(SUITE_NAME, problem_index);
    problem = deprecated__coco_problem_add_observer(problem, OBSERVER_NAME, OBSERVER_OPTIONS);
    if (problem == NULL) {
      printf("problem with index %ld not found, skipped", problem_index);
      continue;
    }
    coco_optimize(problem); /* depending on verbosity, this gives a message from the observer */
    coco_problem_free(problem);  
  }
  printf("Done with suite '%s' (options '%s')", SUITE_NAME, SUITE_OPTIONS);
  if (NUMBER_OF_BATCHES > 1) printf(" batch %d/%d.\n", CURRENT_BATCH, NUMBER_OF_BATCHES);
  else printf(".\n");
  return 0;
}

#elif 0
/* Interface via dimension, function-ID and instance-ID. This does not translate
   directly to different languages or benchmark suites. */
int main(void) {
  long problem_index, instance_id;
  int function_id, dimension_idx;
  coco_problem_t * problem;
  
  /*int *functions;
  int dimensions[] = {2,3,5,10,20,40};
  int *instances;*/
  for (dimension_idx = 0; dimension_idx < 6; dimension_idx++) {/*TODO: find way of using the constants in bbob200_suite*/
      for (function_id = 0; function_id < 24; function_id++) {
          for (instance_id = 0; instance_id < 15; instance_id++) { /* this is specific to 2009 */
              problem_index = bbob2009_encode_problem_index(function_id, instance_id, dimension_idx);
              problem = coco_suite_get_problem(SUITE_NAME, problem_index);
              problem = deprecated__coco_problem_add_observer(OBSERVER_NAME, problem, OBSERVER_OPTIONS);
              if (problem == NULL)
                break;
              coco_optimize(problem);
              coco_problem_free(problem);
          }
      }
    /*bbob2009_get_problem_index(functions[ifun], dimensions[idim], instances[iinst]); */
  }
  return 0;
}
#endif
