#include <stdlib.h>
#include <stdio.h>

#include "coco.h"

void my_optimizer(coco_problem_t *problem) {
  int i;
  static const int budget = 100; /*1000000;*/
  const size_t number_of_variables = coco_get_number_of_variables(problem);
  coco_random_state_t *rng = coco_new_random(0xdeadbeef);
  double *x = coco_allocate_vector(number_of_variables);
  const double *lower = coco_get_smallest_values_of_interest(problem);
  const double *upper = coco_get_largest_values_of_interest(problem);
  double y;

  /* Skip any problems with more than 20 variables */
  /*    if (number_of_variables > 20)
          return;*/
  for (i = 0; i < budget; ++i) {
    size_t j;
    for (j = 0; j < number_of_variables; ++j) {
      const double range = upper[j] - lower[j];
      x[j] = lower[j] + coco_uniform_random(rng) * range;
    }
    coco_evaluate_function(problem, x, &y);
  }
  coco_free_random(rng);
  coco_free_memory(x);
}

/**
 * Return the ${problem_index}-th benchmark problem from the BBOB2009
 * benchmark suit with logging. If the problem index is out of bounds,
 * return NULL.
 *
 * Details: this is rather obsolete?
 */
coco_problem_t *get_bbob2009_problem(const int problem_index,
                                    const char *options) {
  coco_problem_t *problem;
  problem = coco_get_problem("bbob2009", problem_index);
  /* problem = bbob2009_suit(problem_index); */
  if (problem == NULL)  /* this is not necessary anymore */
      return problem;
  problem = coco_observe_problem("bbob2009_observer", problem, options);
  return problem;
}

#if 0
int main() {
  coco_benchmark("bbob2009", "bbob2009_observer", "random_search",
                 my_optimizer);
  return 0;
}
#elif 0
int main() {
  int problem_index; 
  coco_problem_t * problem;
  for (problem_index = 0; ; ++problem_index) {
    /* here we can reject an index, e.g. to distribute the work */
    /* e.g. if (((problem_index + 0) % 5) == 0) */
    problem = get_bbob2009_problem(problem_index, "random_search");
    if (problem == NULL)
      break;
    my_optimizer(problem);
    printf("done with problem %d (function %d)\n",
           problem_index, bbob2009_get_function_id(problem));
    coco_free_problem(problem);
  }
  return 0;
}
#elif 1
int main() {
  const char * suit_name = "bbob2009";
  const char * suit_options = ""; /* e.g.: "instances:1-5, dimensions:-20" */
  const char * observer_name = "bbob2009_observer";
  const char * observer_options = "random_search"; /* future: "folder:random_search, verbosity:1" */
  
  coco_problem_t * problem;
  int problem_index = -1; /* next(-1) == first */
  
  for (problem_index = coco_next_problem_index(suit_name, -1, suit_options); /* next(-1) == first */
       problem_index >= 0;
       problem_index = coco_next_problem_index(suit_name, problem_index, suit_options)
      ) {
      problem = coco_get_problem(suit_name, problem_index);
          /* the following should give a console message by the observer (depending on verbosity): */
      problem = coco_observe_problem(observer_name, problem, observer_options);
          printf("on problem with index %d ... ", problem_index); /* to be removed */
      my_optimizer(problem);
          printf("done\n"); /* to be removed */
      coco_free_problem(problem);  /* this should give a console message by the observer */
  }
  printf("Done with suit %s (options '%s').\n", suit_name, suit_options);
  return 0;
}
#elif 1
/* Interface via dimension, function-ID and instance-ID. This does not translate
   directly to different languages. */
int main() {
  int problem_index, function_id, instance_id, dimension_idx;
  coco_problem_t * problem;
  
  /*int *functions;
  int dimensions[] = {2,3,5,10,20,40};
  int *instances;*/
  for (dimension_idx = 0; dimension_idx < 6; dimension_idx++) {/*TODO: find way of using the constants in bbob200_suit*/
      for (function_id = 0; function_id < 24; function_id++) {
          for (instance_id = 0; instance_id < 15; instance_id++) { /* this is specific to 2009 */
              problem_index = bbob2009_encode_problem_index(function_id, instance_id, dimension_idx);
              problem = get_bbob2009_problem(problem_index, "random_search");
              if (problem == NULL)
              break;
              my_optimizer(problem);
              printf("done with problem %d (function %d)\n",
                     problem_index, bbob2009_get_function_id(problem));
              coco_free_problem(problem);
          }
      }
    /*bbob2009_get_problem_index(functions[ifun], dimensions[idim], instances[iinst]); */
  }
  return 0;
}
#endif
