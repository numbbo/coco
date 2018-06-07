/**
 * Tests that some of the bbob-mixint problems with dimension 2 have been correctly discretized.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "coco.h"
#include "coco.c"

/* Tests whether the transform_vars_discretize was implemented correctly by checking if the
 * optima of the continuous and mixint problems match. This is not a comprehensive test, because it
 * disregards problems with dimension > 2 and those that have more than 20,000 discrete points and
 * checks only the optima (even if the optima match, the transformation could be wrong).
 */
void run_once(char *suite_options) {

  coco_suite_t *suite;
  coco_problem_t *problem_cont, *problem_disc;
  size_t j, k;
  double *xopt_cont, *fopt_cont, fopt_disc;
  double *x, *f;
  size_t num_points[2];
  size_t matching_roi, counter = 0;
  const size_t MAX_NUM = 20000;

  x = coco_allocate_vector(2);
  f = coco_allocate_vector(1);

  suite = coco_suite("bbob-mixint", NULL, suite_options);

  while ((problem_disc = coco_suite_get_next_problem(suite, NULL)) != NULL) {

    /* Skip problems of dimension > 2, those with more than MAX_NUM discrete points or those that
     * whose continuous ROIs do not match */
    if (problem_disc->number_of_variables != 2)
      continue;
    num_points[0] = 1;
    num_points[1] = 1;
    matching_roi = 1;
    for (j = 0; j < 2; j++) {
      if (j < problem_disc->number_of_integer_variables) {
        num_points[j] = (unsigned long)(problem_disc->largest_values_of_interest[j] - problem_disc->smallest_values_of_interest[j] + 1);
      }
      else
        matching_roi = matching_roi && coco_double_almost_equal(problem_disc->largest_values_of_interest[j],
          problem_disc->largest_values_of_interest[j], 1e-10);
    }
    if ((num_points[0] * num_points[1] > MAX_NUM) || (!matching_roi))
      continue;

    /* Compute and compare the optima */
    problem_cont = coco_problem_transformed_get_inner_problem(problem_disc);
    xopt_cont = problem_cont->best_parameter;
    fopt_cont = problem_cont->best_value;
    fopt_disc = DBL_MAX;
    for (j = 0; j < num_points[0]; j++) {
      x[0] = ((num_points[0] == 1) ? xopt_cont[0] : (double)j);
      for (k = 0; k < num_points[1]; k++) {
        x[1] = ((num_points[1] == 1) ? xopt_cont[1] : (double)k);
        coco_evaluate_function(problem_disc, x, f);
        if (f[0] < fopt_disc) {
          fopt_disc = f[0];
        }
      }
    }
    if (!coco_double_almost_equal(fopt_cont[0], fopt_disc, 1e-10)) {
      coco_suite_free(suite);
      coco_free_memory(x);
      coco_free_memory(f);
      coco_error("The optima of the original and discretized problem %lu do (%f, %f) not match!",
          (unsigned long)coco_problem_get_suite_dep_index(problem_disc), fopt_cont[0], fopt_disc);
    }
    counter++;
  }

  coco_suite_free(suite);
  coco_free_memory(x);
  coco_free_memory(f);

  if (counter == 0)
    coco_error("No tests of bbob-mixint were performed!");
  else
    printf("Performed %lu tests on bbob-mixint\n", (unsigned long)counter);
  printf("DONE!\n");
  fflush(stdout);
}

int main(void)  {

  run_once("dimensions: 2");

  coco_remove_directory("exdata");
  return 0;
}
