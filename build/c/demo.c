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
static const int MAX_BUDGET = 1e3;  /* work on small budgets first */
static coco_problem_t *current_coco_problem; /* used in objective_function */

/**************************************************
 * Example objective function interface and solver,
 * no need to modify 
 **************************************************/
typedef double (*objective_function_example_t) (const double *);
double objective_function_example(const double *x) {
    double y;
    coco_evaluate_function(current_coco_problem, x, &y);
    return y;
}
void random_search(const size_t dimension,
                   const objective_function_example_t fun,
                   const double *lower,
                   const double *upper) {
  coco_random_state_t *rng = coco_new_random(0xdeadbeef); /* use coco fcts for convenience */
  double *x = coco_allocate_vector(dimension);
  double y;
  int i;

  for (i = 0; i < MAX_BUDGET; ++i) {
    size_t j;
    for (j = 0; j < dimension; ++j) {
      x[j] = lower[j] + coco_uniform_random(rng) * (upper[j] - lower[j]);
    }
    y = fun(x);

    /* To be of any real use, we would need to retain the best x-value here.
     * For benchmarking purpose the implementation suffices, as the
     * observer takes care of bookkeeping. */
  }
  coco_free_random(rng);
  coco_free_memory(x);
}

/**************************************************
 *   Set up the experiment                    
 **************************************************/
static const char * suite_name       = "bbob2009";
static const char * suite_options    = ""; /* e.g.: "instances:1-5; dimensions:-20" */
static const char * observer_name    = "bbob2009_observer"; /* writes data */
/* static const char * observer_name = "no_observer"; / * writes no data */
static const char * observer_options = "random_search_on_bbob2009"; /* future: "folder:random_search; verbosity:1" */
static const char * solver_name      = "random_search"; /* for the choice in coco_solver below */
/*  static const char * solver_name   = "my_solver"; / * for the choice in coco_solver below */
static const int number_of_batches   = 1;  /* use 1 for single batch :-) batches can be run independently in parallel */
static int current_batch             = 1;  /* runs from 1 to number_of_batches, or any other consecutive sequence */

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
  coco_evaluate_function(current_coco_problem, x, y); /* this call writes objective_function(x) in y */
}
/* Minimal solver definition only to avoid compile/link errors below for my_solver */
void my_solver(objective_function_t func,
               const double *initial_x, size_t dim) {
  double y; func(initial_x, &y); /* rather useless solver evaluates only the initial value */
}
/**
 * Finally, coco_solver calls, depending on solver_name, one
 * of the defined solvers (e.g. random_search, my_solver, ...),
 * using the respective matching objective function. 
 */
void coco_solver(coco_problem_t *problem) {
  /* prepare, set up convenience definitions */
  size_t dimension = coco_get_number_of_variables(problem);
  const double * lbounds = coco_get_smallest_values_of_interest(problem);
  const double * ubounds = coco_get_largest_values_of_interest(problem);
  double * initial_x = coco_allocate_vector(coco_get_number_of_variables(problem));
  coco_get_initial_solution(problem, initial_x);
  current_coco_problem = problem; /* do not change this, it's used in objective_function */

  /* ENHANCE: loop over independent restarts here */
  /* call the solver */
  if (strcmp("random_search", solver_name) == 0) { /* example case, no need to modify */
    random_search(dimension, objective_function_example,
                  lbounds, ubounds);
  } else if (strcmp("my_solver", solver_name) == 0) {
    /* MODIFY this call according to objective_function_t and solver's needs */
    my_solver(objective_function, /* MODIFY my_solver to your_solver ... */
              initial_x, dimension); 
  }
  coco_free_memory(initial_x);
}

/**************************************************
 *   run the experiment                       
 **************************************************/
#if 0
int main() {
  coco_benchmark(suite_name, observer_name, observer_options,
                 coco_solver);
  return 0;
}

#elif 1
int main() {
  coco_problem_t * problem;
  int problem_index = coco_next_problem_index(suite_name, -1,
                                              suite_options); /* next(-1) == first */
  if (number_of_batches > 1)
    printf("Running only batch %d out of %d batches for suite %s\n",
           current_batch, number_of_batches, suite_name);
  for ( ; problem_index >= 0;
       problem_index = coco_next_problem_index(suite_name, problem_index, suite_options)
      ) {
    /* here we reject indices from other batches */
    if (((problem_index - current_batch + 1) % number_of_batches) != 0)
           continue;
    problem = coco_get_problem(suite_name, problem_index);
    problem = coco_observe_problem(observer_name, problem, observer_options);
    if (problem == NULL) {
      printf("problem with index %d not found, skipped", problem_index);
      continue;
    }
    coco_solver(problem);
    coco_free_problem(problem);  /* this should give a console message by the observer */
  }
  printf("Done with suite '%s' (options '%s')", suite_name, suite_options);
  if (number_of_batches > 1) printf(" batch %d/%d.\n", current_batch, number_of_batches);
  else printf(".\n");
  return 0;
}

#elif 1
/* Interface via dimension, function-ID and instance-ID. This does not translate
   directly to different languages or benchmark suites. */
static const coco_optimizer_t coco_solver = coco_random_search;
int main() {
  int problem_index, function_id, instance_id, dimension_idx;
  coco_problem_t * problem;
  
  /*int *functions;
  int dimensions[] = {2,3,5,10,20,40};
  int *instances;*/
  for (dimension_idx = 0; dimension_idx < 6; dimension_idx++) {/*TODO: find way of using the constants in bbob200_suite*/
      for (function_id = 0; function_id < 24; function_id++) {
          for (instance_id = 0; instance_id < 15; instance_id++) { /* this is specific to 2009 */
              problem_index = bbob2009_encode_problem_index(function_id, instance_id, dimension_idx);
              problem = coco_get_problem(suite_name, problem_index);
              problem = coco_observe_problem(observer_name, problem, observer_options);
              if (problem == NULL)
                break;
              coco_solver(problem);
              // printf("done with problem %d (function %d)\n",
              //        problem_index, bbob2009_get_function_id(problem));
              coco_free_problem(problem);
          }
      }
    /*bbob2009_get_problem_index(functions[ifun], dimensions[idim], instances[iinst]); */
  }
  return 0;
}
#endif
