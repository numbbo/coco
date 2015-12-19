#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob2009_legacy_code.c"
#include "transform_obj_shift.c"

static double f_linear_slope_raw(const double *x,
                                 const size_t number_of_variables,
                                 const double *best_parameter) {

  static const double alpha = 100.0;
  size_t i;
  double result = 0.0;

  for (i = 0; i < number_of_variables; ++i) {
    double base, exponent, si;

    base = sqrt(alpha);
    exponent = (double) (long) i / ((double) (long) number_of_variables - 1);
    if (best_parameter[i] > 0.0) {
      si = pow(base, exponent);
    } else {
      si = -pow(base, exponent);
    }
    result += 5.0 * fabs(si) - si * x[i];
  }

  return result;
}

static void f_linear_slope_evaluate(coco_problem_t *self, const double *x, double *y) {
  assert(self->number_of_objectives == 1);
  y[0] = f_linear_slope_raw(x, self->number_of_variables, self->best_parameter);
}

static coco_problem_t *f_linear_slope_allocate(const size_t number_of_variables, const double *best_parameter) {

  size_t i;
  /* best_parameter will be overwritten below */
  coco_problem_t *problem = coco_problem_allocate_from_scalars("linear slope function",
      f_linear_slope_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "linear_slope", number_of_variables);

  /* Compute best solution */
  for (i = 0; i < number_of_variables; ++i) {
    if (best_parameter[i] < 0.0) {
      problem->best_parameter[i] = problem->smallest_values_of_interest[i];
    } else {
      problem->best_parameter[i] = problem->largest_values_of_interest[i];
    }
  }
  f_linear_slope_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

static coco_problem_t *f_linear_slope_bbob_problem_allocate(const size_t function,
                                                            const size_t dimension,
                                                            const size_t instance,
                                                            const long rseed,
                                                            const char *problem_id_template,
                                                            const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  fopt = bbob2009_compute_fopt(function, instance);

  problem = f_linear_slope_allocate(dimension, xopt);
  problem = f_transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

  coco_free_memory(xopt);
  return problem;
}

/* TODO: Deprecated functions below are to be deleted when the new ones work as they should */

static void deprecated__f_linear_slope_evaluate(coco_problem_t *self, const double *x, double *y) {
  static const double alpha = 100.0;
  size_t i;
  assert(self->number_of_objectives == 1);
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    double base, exponent, si;

    base = sqrt(alpha);
    exponent = (double) (long) i / ((double) (long) self->number_of_variables - 1);
    if (self->best_parameter[i] > 0.0) {
      si = pow(base, exponent);
    } else {
      si = -pow(base, exponent);
    }
    y[0] += 5.0 * fabs(si) - si * x[i];
  }
}

static coco_problem_t *deprecated__f_linear_slope(const size_t number_of_variables, const double *best_parameter) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_problem_allocate(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("linear slope function");
  /* Construct a meaningful problem id */
  problem_id_length = (size_t) snprintf(NULL, 0, "%s_%02lu", "linear_slope", (long) number_of_variables);
  problem->problem_id = (char *) coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02lu", "linear_slope",
      (long) number_of_variables);

  problem->evaluate_function = deprecated__f_linear_slope_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    if (best_parameter[i] < 0.0) {
      problem->best_parameter[i] = problem->smallest_values_of_interest[i];
    } else {
      problem->best_parameter[i] = problem->largest_values_of_interest[i];
    }
  }
  /* Calculate best parameter value */
  deprecated__f_linear_slope_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
