#include <stdlib.h>
#include <stdio.h>

#include "coco.h"

void my_optimizer(coco_problem_t *problem) {
  static const int budget = 102; /* 100000;*/
  coco_random_state_t *rng = coco_random_new(0xdeadbeef);

  double *x = (double *)malloc(coco_problem_get_dimension(problem) * sizeof(double));
  double y;
  for (int i = 1; i < budget; ++i) {
    bbob2009_unif(x, coco_problem_get_dimension(problem), i);
    for (int j = 0; j < coco_problem_get_dimension(problem); ++j) {
      /*const double range = problem->upper_bounds[j] -
      problem->lower_bounds[j];
      x[j] = problem->lower_bounds[j] + coco_uniform_random(rng) * range;*/
      x[j] = 20. * x[j] - 10.;
    }
    /*if ( i % 100==0 ){
        printf("%d: [ %f,", i, x[0]);
        printf(" %f ]\n",x[1]);
    }*/
    coco_evaluate_function(problem, x, &y);
  }
  coco_random_free(rng);
  free(x);
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
