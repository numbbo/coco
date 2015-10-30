#include <stdlib.h>
#include <stdio.h>

#include "numbbo.h"
#include "../../src/bbob2009_fopt.c"

void my_optimizer(coco_problem_t *problem) {
  static const int budget = 102; /* 100000;*/
  coco_random_state_t *rng = coco_random_new(0xdeadbeef);
  double *x = (double *)malloc(problem->number_of_parameters * sizeof(double));
  double y;
  FILE *fd = fopen("const_check_newC.txt", "w");
  fputs("x[0] x[1] f(x)\n", fd);
  for (int i = 1; i < budget; ++i) {
    bbob2009_unif(x, problem->number_of_parameters, i);
    for (int j = 0; j < problem->number_of_parameters; ++j) {
      /*const double range = problem->upper_bounds[j] -
       problem->lower_bounds[j];
       x[j] = problem->lower_bounds[j] + coco_uniform_random(rng) * range;*/
      x[j] = 20. * x[j] - 10.;
    }
    coco_evaluate_function(problem, x, &y);
    if (i % 1 == 0) {
      fprintf(fd, "%.5f %.5f : %.5f\n", x[0], x[1], y);
    }
  }
  coco_random_free(rng);
  free(x);
  fclose(fd);
  printf("%s\n", problem->problem_name);
  printf("%s\n", problem->problem_id);
}

int main(int argc, char **argv) {
  coco_suite_benchmark("toy_suit", "toy_observer", "random_search", my_optimizer);
}
/*
 params.funcId = ifun;
 params.DIM = dim[idx_dim];
 params.instanceId = instance;
 printf("%d-D f%d, trial: %d\n", dim[idx_dim], ifun, instance);
 fgeneric_initialize(params);
 for (i = 1; i <= 1001; i++)
 {
 unif(indiv, dim[idx_dim], i);
 for (j = 0; j < dim[idx_dim]; j++)
 indiv[j] = 20. * indiv[j] - 10.;
 fgeneric_evaluate(indiv);
 }
 fgeneric_finalize();
 printf("\tDone, elapsed time [h]: %.2f\n",
 (double)(clock()-t0)/CLOCKS_PER_SEC/60./60.);
 */
