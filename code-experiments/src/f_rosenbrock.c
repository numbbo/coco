#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob2009_legacy_code.c"
#include "transform_vars_shift.c"
#include "transform_vars_scale.c"
#include "transform_vars_affine.c"
#include "transform_obj_shift.c"

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

static coco_problem_t *f_rosenbrock_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Rosenbrock function",
      f_rosenbrock_evaluate, NULL, number_of_variables, -5.0, 5.0, 1.0);
  coco_problem_set_id(problem, "%s_d%04lu", "rosenbrock", number_of_variables);

  /* Compute best solution */
  f_rosenbrock_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

static coco_problem_t *f_rosenbrock_bbob_problem_allocate(const size_t function_id,
                                                          const size_t dimension,
                                                          const size_t instance_id,
                                                          const long rseed,
                                                          const char *problem_id_template,
                                                          const char *problem_name_template) {

  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  size_t i;
  double *minus_one, factor;

  minus_one = coco_allocate_vector(dimension);
  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  for (i = 0; i < dimension; ++i) {
    minus_one[i] = -1.0;
    xopt[i] *= 0.75;
  }
  fopt = bbob2009_compute_fopt(function_id, instance_id);
  factor = coco_max_double(1.0, sqrt((double) dimension) / 8.0);

  problem = f_rosenbrock_allocate(dimension);
  problem = f_transform_vars_shift(problem, minus_one, 0);
  problem = f_transform_vars_scale(problem, factor);
  problem = f_transform_vars_shift(problem, xopt, 0);
  problem = f_transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function_id, instance_id, dimension);
  coco_problem_set_name(problem, problem_name_template, function_id, instance_id, dimension);

  coco_free_memory(minus_one);
  coco_free_memory(xopt);
  return problem;
}

static coco_problem_t *f_rosenbrock_rotated_bbob_problem_allocate(const size_t function_id,
                                                                  const size_t dimension,
                                                                  const size_t instance_id,
                                                                  const long rseed,
                                                                  const char *problem_id_template,
                                                                  const char *problem_name_template) {

  double fopt;
  coco_problem_t *problem = NULL;
  size_t row, column;
  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *current_row, **rot1, factor;

  fopt = bbob2009_compute_fopt(function_id, instance_id);
  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed, dimension);

  factor = coco_max_double(1.0, sqrt((double) dimension) / 8.0);
  /* Compute affine transformation */
  for (row = 0; row < dimension; ++row) {
    current_row = M + row * dimension;
    for (column = 0; column < dimension; ++column) {
      current_row[column] = rot1[row][column];
      if (row == column)
        current_row[column] *= factor;
    }
    b[row] = 0.5;
  }
  bbob2009_free_matrix(rot1, dimension);

  problem = f_rosenbrock_allocate(dimension);
  problem = f_transform_vars_affine(problem, M, b, dimension);
  problem = f_transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function_id, instance_id, dimension);
  coco_problem_set_name(problem, problem_name_template, function_id, instance_id, dimension);

  coco_free_memory(M);
  coco_free_memory(b);
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
