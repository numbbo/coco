#include <stdlib.h>
#include <stdio.h>

#include "coco.h"

static const int MAX_BUDGET = 100;  /* work on small budgets first */
static coco_problem_t *current_coco_problem; /* used in objective_function */

/*** example solver, no need to modify ***/
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
    /* to be of any real use, we would need to retain the best x-value here */
  }
  coco_free_random(rng);
  coco_free_memory(x);
}

/*** set up solver interface ***/
/* Definition of the objective function must probably
 * be adapted to the solver used. 
 * */
typedef double (*objective_function_t) (const double *); /* change if necessary */
double objective_function(const double *x) {
    double y;
    coco_evaluate_function(current_coco_problem, x, &y);
    return y;
}
/** Calling the "generic" solver (here random_search) in coco_solver. 
 */
void coco_solver(coco_problem_t *problem) {
  /* convenience definitions */
  size_t dimension = coco_get_number_of_variables(problem);
  const double * lbounds = coco_get_smallest_values_of_interest(problem);
  const double * ubounds = coco_get_largest_values_of_interest(problem);

  current_coco_problem = problem; /* do not change this, it's used in objective_function */

  if (0) /* example case, no need to modify */
    random_search(dimension,
                objective_function_example,
                lbounds,
                ubounds);
  else { /* modify as necessary according to objective_function_t and solvers needs */
    random_search(dimension, 
                objective_function,
                coco_get_smallest_values_of_interest(problem),
                coco_get_largest_values_of_interest(problem));
  }
}

/*** set up the experiment ***/
static const char * suite_name = "bbob2009";
static const char * suite_options = ""; /* e.g.: "instances:1-5; dimensions:-20" */
static const char * observer_name = "bbob2009_observer"; /* writes the data */
static const char * observer_options = "random_search_on_bbob2009"; /* future: "folder:random_search; verbosity:1" */

/* run the experiment */
#if 0
int main() {
  coco_benchmark(suite_name, observer_name, observer_options,
                 coco_solver);
  return 0;
}

#elif 1
int main() {
  coco_problem_t * problem;
  int problem_index = coco_next_problem_index(suite_name, -1, suite_options); /* next(-1) == first */
  
  for ( ; problem_index >= 0;
       problem_index = coco_next_problem_index(suite_name, problem_index, suite_options)
      ) {
    /* here we could reject an index, e.g. to distribute the work, e.g. */
    /* if (((problem_index + 0) % 5) != 0)
           continue;
    */
    problem = coco_get_problem(suite_name, problem_index);
        /* the following should give a console message by the observer (depending on verbosity): */
    problem = coco_observe_problem(observer_name, problem, observer_options);
    if (problem == NULL) {
      printf("problem with index %d not found, skipped", problem_index);
      continue;
    }
    coco_solver(problem);
    coco_free_problem(problem);  /* this should give a console message by the observer */
  }
  printf("Done with suite %s (options '%s').\n", suite_name, suite_options);
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
