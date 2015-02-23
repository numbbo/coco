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
 */
coco_problem_t *get_bbob2009_problem(const int problem_index,
                                    const char *options) {
  coco_problem_t *problem;
  problem = coco_get_problem("bbob2009", problem_index);
  /* problem = bbob2009_suit(problem_index); */
  if (problem == NULL)
      return problem;
  problem = coco_observe_problem("bbob2009_observer", problem, options);
  return problem;
}

#if 11 < 3
int main() {
  coco_benchmark("bbob2009", "bbob2009_observer", "random_search",
                 my_optimizer);
  return 0;
}
#elif 11 < 3
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
#else
int main() {
  int problem_index, function_id, instance_id, dimension_idx;
  coco_problem_t * problem;
  /*int *functions;
  int dimensions[] = {2,3,5,10,20,40};
  int *instances;*/
  for (dimension_idx = 0; dimension_idx < 6; dimension_idx++) {/*TODO: find way of using the constants in bbob200_suit*/
      for (function_id = 0; function_id < 24; function_id++) {
          for (instance_id = 0; instance_id < 15; instance_id++) {
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