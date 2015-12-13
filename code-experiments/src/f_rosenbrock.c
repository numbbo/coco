#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

static double f_rosenbrock_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double result;
  double s1 = 0.0, s2 = 0.0, tmp;

  assert(number_of_variables > 1);

  for (i = 0; i < number_of_variables - 1; ++i) {
    tmp = (x[i] * x[i] - x[i + 1]);
    s1 += tmp * tmp;
    tmp = (x[i] - 1.0);
    s2 += tmp * tmp;
  }
  result = 100.0 * s1 + s2;

  return result;
}

static void f_rosenbrock_evaluate(coco_problem_t *self, const double *x, double *y) {
  assert(self->number_of_objectives == 1);
  y[0] = f_rosenbrock_raw(x, self->number_of_variables);
}

static coco_problem_t *f_rosenbrock(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Rosenbrock function",
      f_rosenbrock_evaluate, NULL, number_of_variables, -5.0, 5.0, 1.0);
  coco_problem_set_id(problem, "%s_d%04lu", "rosenbrock", number_of_variables);

  /* Compute best solution */
  f_rosenbrock_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/* TODO: Deprecated functions below are to be deleted when the new ones work as they should */

static void deprecated__f_rosenbrock_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  double s1 = 0.0, s2 = 0.0, tmp;
  assert(self->number_of_objectives == 1);
  assert(self->number_of_variables > 1);
  for (i = 0; i < self->number_of_variables - 1; ++i) {
    tmp = (x[i] * x[i] - x[i + 1]);
    s1 += tmp * tmp;
    tmp = (x[i] - 1.0);
    s2 += tmp * tmp;
  }
  y[0] = 100.0 * s1 + s2;
}

static coco_problem_t *deprecated__f_rosenbrock(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_problem_allocate(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("rosenbrock function");
  /* Construct a meaningful problem id */
  problem_id_length = (size_t) snprintf(NULL, 0, "%s_%02lu", "rosenbrock", (long) number_of_variables);
  problem->problem_id = (char *) coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02lu", "rosenbrock", (long) number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = deprecated__f_rosenbrock_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 1.0;
  }
  /* Calculate best parameter value */
  deprecated__f_rosenbrock_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
