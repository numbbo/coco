#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "coco.h"

#include "coco_problem.c"

static void f_griewankRosenbrock_evaluate(coco_problem_t *self, double *x,
                                          double *y) {
  size_t i;
  double tmp = 0;
  assert(self->number_of_objectives == 1);

  /* Computation core */
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables - 1; ++i) {
    const double c1 = x[i] * x[i] - x[i + 1];
    const double c2 = 1.0 - x[i];
    tmp = 100.0 * c1 * c1 + c2 * c2;
    y[0] += tmp / 4000. - cos(tmp);
  }
  y[0] = 10. + 10. * y[0] / (double)(self->number_of_variables - 1);
}

static coco_problem_t *
griewankRosenbrock_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("griewank rosenbrock function");
  /* Construct a meaningful problem id */
  problem_id_length = snprintf(NULL, 0, "%s_%02i", "griewank rosenbrock",
                               (int)number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d",
           "griewank rosenbrock", (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = f_griewankRosenbrock_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 1.0; /* z^opt = 1*/
  }
  /* Calculate best parameter value */
  f_griewankRosenbrock_evaluate(problem, problem->best_parameter,
                                problem->best_value);
  return problem;
}
