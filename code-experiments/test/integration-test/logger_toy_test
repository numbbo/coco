#include <stdlib.h>
#include <stdio.h>

#include "coco.h"

const char file_name[200] = {"RS_on_suite_toy/test_fun_constistency.txt"};

/**
 * A random search optimizer.
 */
void my_optimizer(coco_problem_t *problem) {

  const size_t budget = 101;
  coco_random_state_t *rng = coco_random_new(0xdeadbeef);
  const double *lbounds = coco_problem_get_smallest_values_of_interest(problem);
  const double *ubounds = coco_problem_get_largest_values_of_interest(problem);
  size_t dimension = coco_problem_get_dimension(problem);
  double *x = coco_allocate_vector(dimension);
  double y, range;
  size_t i, j;

  FILE *fd = fopen(file_name, "a");

  fprintf(fd, "%s, %s\n", coco_problem_get_id(problem), coco_problem_get_name(problem));
  fputs("eval f(x) x[0] x[1]\n", fd);

  for (i = 1; i < budget; ++i) {

    for (j = 0; j < dimension; ++j) {
      range = ubounds[j] - lbounds[j];
      x[j] = lbounds[j] + coco_random_uniform(rng) * range;
    }
    coco_evaluate_function(problem, x, &y);

    fprintf(fd, "%5lu\t%11.5f\t%11.5f\t%11.5f\n", i, y, x[0], x[1]);
    fflush(fd);

  }
  fputs("\n", fd);
  fclose(fd);
  coco_random_free(rng);
  coco_free_memory(x);
}

int main(void) {
  /* Remove any data left from previous runs */
  coco_remove_directory("RS_on_suite_toy");

  /* Run the benchmark */
  coco_suite_benchmark("suite_toy", "observer_toy", "result_folder: RS_on_suite_toy", my_optimizer);
  return 0;
}
