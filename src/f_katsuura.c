#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "coco.h"

#include "coco_problem.c"

static void f_katsuura_evaluate(coco_problem_t *self, double *x, double *y) {
  size_t i, j;
  double tmp, tmp2;
  assert(self->number_of_objectives == 1);

  /* Computation core */
  y[0] = 1.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    tmp = 0;
    for (j = 1; j < 33; ++j) {
      tmp2 = pow(2., (double)j);
      tmp += fabs(tmp2 * x[i] - round(tmp2 * x[i])) / tmp2;
    }
    tmp = 1. + (i + 1) * tmp;
    y[0] *= tmp;
  }
  y[0] = 10. / ((double)self->number_of_variables) /
         ((double)self->number_of_variables) *
         (-1. + pow(y[0], 10. / pow((double)self->number_of_variables, 1.2)));
}

static coco_problem_t *katsuura_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("katsuura function");
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "katsuura", (int)number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "katsuura",
           (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = f_katsuura_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 1.0; /* z^opt = 1*/
  }
  /* Calculate best parameter value */
  f_katsuura_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
