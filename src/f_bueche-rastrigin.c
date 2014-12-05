#include <math.h>
#include <assert.h>

#include "coco.h"

#include "coco_problem.c"

static void _bueche_rastrigin_evaluate(coco_problem_t *self, double *x,
                                       double *y) {
  size_t i;
  double tmp = 0., tmp2 = 0.;
  assert(self->number_of_objectives == 1);
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    tmp += cos(2 * coco_pi * x[i]);
    tmp2 += x[i] * x[i];
  }
  y[0] = 10 * (self->number_of_variables - tmp) + tmp2 + 0;
}

static coco_problem_t *
bueche_rastrigin_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("Bueche-Rastrigin function");
  /* Construct a meaningful problem id */
  problem_id_length = snprintf(NULL, 0, "%s_%02i", "bueche-rastrigin",
                               (int)number_of_variables);
  problem->problem_id = (char *)coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d",
           "skewRastriginBueche", (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = _bueche_rastrigin_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }
  /* Calculate best parameter value */
  _bueche_rastrigin_evaluate(problem, problem->best_parameter,
                             problem->best_value);
  return problem;
}
