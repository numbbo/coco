#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "cmaes_interface.h"
#include "coco.h"

static int function_id = 0;

static bool is_feasible(const double *x, const double *lower,
                        const double *upper, size_t number_of_variables) {
  for (size_t i = 0; i < number_of_variables; ++i) {
    if (x[i] < lower[i] || x[i] > upper[i]) {
      return false;
    }
  }
  return true;
}

void cma_optimizer(coco_problem_t *problem) {
  cmaes_t cma;
  double *const *X, *y;

  /* Extract number of variables and bounds from problem */
  const size_t number_of_variables = coco_get_number_of_variables(problem);
  const double *lower = coco_get_smallest_values_of_interest(problem);
  const double *upper = coco_get_largest_values_of_interest(problem);

  /* Skip "big" problems for now... */
  if (number_of_variables > 20) {
    fprintf(stderr, "fid-%04i: Skipping '%s'\n", function_id++,
        coco_get_problem_id(problem));
    return;
  } else {
    fprintf(stderr, "fid-%04i: Optimizing '%s'\n", function_id++,
            coco_get_problem_id(problem));
  }

  /* Allocate vectors for initial solution and initial sigma */
  double *initial_solution = coco_allocate_vector(number_of_variables);
  double *initial_sigma = coco_allocate_vector(number_of_variables);

  for (size_t i = 0; i < number_of_variables; ++i) {
    initial_solution[i] = lower[i] + (upper[i] - lower[i]) / 2.0;
    /* Based on the +/-3sigma rule to obtain a 99.7% CI */
    initial_sigma[i] = (upper[i] - lower[i]) / 6.0;
  }

  /* Initialize CMA-ES without parameter file */
  cmaes_init_para(&cma, number_of_variables, initial_solution, initial_sigma, 0,
                  100, "no");
  cma.sp.filename = strdup("no");
  y = cmaes_init_final(&cma);

  /* Main evolutionary loop */
  while (!cmaes_TestForTermination(&cma)) {
    X = cmaes_SamplePopulation(&cma);

    for (size_t i = 0; i < cmaes_Get(&cma, "lambda"); ++i) {
      /* Make sure candidate solution is inside the problems bounding box. If
       * not, resample until it is.
       */
      while (!is_feasible(X[i], lower, upper, number_of_variables)) {
        cmaes_ReSampleSingle(&cma, i);
      }
      coco_evaluate_function(problem, X[i], &y[i]);
    }
    cmaes_UpdateDistribution(&cma, y);
  }

  /* Cleanup and free memory */
  cmaes_exit(&cma);
  coco_free_memory(initial_solution);
  coco_free_memory(initial_sigma);
}

int main() {
  char algorithm_id[200] = "c-cmaes-v";

  /* Extract cmaes library version. */
  cmaes_t cma;
  cmaes_init(&cma, 1, NULL, NULL, 0, 100, "no");
  strncat(algorithm_id, cma.version, 200 - strlen(algorithm_id) - 1);
  cmaes_exit(&cma);

  coco_benchmark("bbob2009", "bbob2009_observer", algorithm_id, cma_optimizer);
  return 0;
}
