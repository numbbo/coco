#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "coco_generics.c"
#include "bbob2009_legacy_code.c"

static void f_schwefel_evaluate(coco_problem_t *self, double *x, double *y) {
  size_t i;
  double penalty, sum;
  assert(self->number_of_objectives == 1);

  /* Boundary handling*/
  penalty = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    const double tmp = fabs(x[i]) - 500.0;
    if (tmp > 0.0)
      penalty += tmp * tmp;
  }

  /* Computation core */
  sum = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    sum += x[i] * sin(sqrt(fabs(x[i])));
  }
  y[0] = 0.01 * (penalty + 418.9828872724339 -
                 sum / (double)self->number_of_variables);
}

static coco_problem_t *schwefel_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  /*fprintf(stdout, "%2i\n",
              instance);
  fflush(stdout);*/
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("schwefel function");
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "schwefel", (int)number_of_variables);
  problem->problem_id = (char *)coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "schwefel",
           (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = f_schwefel_evaluate;

  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0;
  }
  /* Calculate best parameter value */
  f_schwefel_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
