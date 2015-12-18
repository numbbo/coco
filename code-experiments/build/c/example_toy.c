#include <stdlib.h>
#include <stdio.h>

#include "coco.h"

/**
 * A random search optimizer.
 */
void my_random_search(coco_problem_t *problem) {

  const size_t budget = 101;
  coco_random_state_t *rng = coco_random_new(0xdeadbeef);
  const double *lbounds = coco_problem_get_smallest_values_of_interest(problem);
  const double *ubounds = coco_problem_get_largest_values_of_interest(problem);
  size_t dimension = coco_problem_get_dimension(problem);
  double *x = coco_allocate_vector(dimension);
  double y, range;
  size_t i, j;

  for (i = 1; i < budget; ++i) {

    for (j = 0; j < dimension; ++j) {
      range = ubounds[j] - lbounds[j];
      x[j] = lbounds[j] + coco_random_uniform(rng) * range;
    }
    coco_evaluate_function(problem, x, &y);

  }

  coco_random_free(rng);
  coco_free_memory(x);
}

int main(void) {
  /* Run the benchmark */
  coco_suite_benchmark("suite_toy", "observer_toy", "result_folder: RS_on_suite_toy", my_random_search);
  return 0;
}
