#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob2009_legacy_code.c"
#include "transform_obj_penalize.c"
#include "transform_obj_shift.c"
#include "transform_vars_affine.c"
#include "transform_vars_oscillate.c"
#include "transform_vars_shift.c"

/* Number of summands in the Weierstrass problem. */
#define WEIERSTRASS_SUMMANDS 12
typedef struct {
  double f0;
  double ak[WEIERSTRASS_SUMMANDS];
  double bk[WEIERSTRASS_SUMMANDS];
} f_weierstrass_data_t;

static double f_weierstrass_raw(const double *x, const size_t number_of_variables, f_weierstrass_data_t *data) {

  size_t i, j;
  double result;

  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    for (j = 0; j < WEIERSTRASS_SUMMANDS; ++j) {
      result += cos(2 * coco_pi * (x[i] + 0.5) * data->bk[j]) * data->ak[j];
    }
  }
  result = 10.0 * pow(result / (double) (long) number_of_variables - data->f0, 3.0);

  return result;
}

static void f_weierstrass_evaluate(coco_problem_t *self, const double *x, double *y) {
  assert(self->number_of_objectives == 1);
  y[0] = f_weierstrass_raw(x, self->number_of_variables, self->data);
}

static coco_problem_t *f_weierstrass_allocate(const size_t number_of_variables) {

  f_weierstrass_data_t *data;
  size_t i;
  double *non_unique_best_value;
  coco_problem_t *problem = coco_problem_allocate_from_scalars("Weierstrass function",
      f_weierstrass_evaluate, NULL, number_of_variables, -5.0, 5.0, NAN);
  coco_problem_set_id(problem, "%s_d%02lu", "weierstrass", number_of_variables);

  data = coco_allocate_memory(sizeof(*data));
  data->f0 = 0.0;
  for (i = 0; i < WEIERSTRASS_SUMMANDS; ++i) {
    data->ak[i] = pow(0.5, (double) i);
    data->bk[i] = pow(3., (double) i);
    data->f0 += data->ak[i] * cos(2 * coco_pi * data->bk[i] * 0.5);
  }
  problem->data = data;

  /* Compute best solution */
  non_unique_best_value = coco_allocate_vector(number_of_variables);
  for (i = 0; i < number_of_variables; i++)
    non_unique_best_value[i] = 0.0;
  f_weierstrass_evaluate(problem, non_unique_best_value, problem->best_value);
  coco_free_memory(non_unique_best_value);
  return problem;
}

static coco_problem_t *f_weierstrass_bbob_problem_allocate(const size_t function,
                                                           const size_t dimension,
                                                           const size_t instance,
                                                           const long rseed,
                                                           const char *problem_id_template,
                                                           const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  size_t i, j, k;
  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *current_row, **rot1, **rot2;

  const double condition = 100.0;
  const double penalty_factor = 10.0 / (double) dimension;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  rot2 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_compute_rotation(rot2, rseed, dimension);
  for (i = 0; i < dimension; ++i) {
    b[i] = 0.0;
    current_row = M + i * dimension;
    for (j = 0; j < dimension; ++j) {
      current_row[j] = 0.0;
      for (k = 0; k < dimension; ++k) {
        const double base = 1.0 / sqrt(condition);
        const double exponent = 1.0 * (int) k / ((double) (long) dimension - 1.0);
        current_row[j] += rot1[i][k] * pow(base, exponent) * rot2[k][j];
      }
    }
  }

  problem = f_weierstrass_allocate(dimension);
  problem = f_transform_obj_shift(problem, fopt);
  problem = f_transform_vars_affine(problem, M, b, dimension);
  problem = f_transform_vars_oscillate(problem);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  problem = f_transform_vars_affine(problem, M, b, dimension);
  problem = f_transform_vars_shift(problem, xopt, 0);
  problem = f_transform_obj_penalize(problem, penalty_factor);

  bbob2009_free_matrix(rot1, dimension);
  bbob2009_free_matrix(rot2, dimension);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "4-multi-modal");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}

/* TODO: Deprecated functions below are to be deleted when the new ones work as they should */

static void deprecated__f_weierstrass_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i, j;
  f_weierstrass_data_t *data = self->data;
  assert(self->number_of_objectives == 1);

  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    for (j = 0; j < WEIERSTRASS_SUMMANDS; ++j) {
      y[0] += cos(2 * coco_pi * (x[i] + 0.5) * data->bk[j]) * data->ak[j];
    }
  }
  y[0] = 10.0 * pow(y[0] / (double) (long) self->number_of_variables - data->f0, 3.0);
}

static coco_problem_t *deprecated__f_weierstrass(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_problem_allocate(number_of_variables, 1, 0);
  f_weierstrass_data_t *data;
  data = coco_allocate_memory(sizeof(*data));

  data->f0 = 0.0;
  for (i = 0; i < WEIERSTRASS_SUMMANDS; ++i) {
    data->ak[i] = pow(0.5, (double) i);
    data->bk[i] = pow(3., (double) i);
    data->f0 += data->ak[i] * cos(2 * coco_pi * data->bk[i] * 0.5);
  }

  problem->problem_name = coco_strdup("weierstrass function");
  /* Construct a meaningful problem id */
  problem_id_length = (size_t) snprintf(NULL, 0, "%s_%02lu", "weierstrass", (long) number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02lu", "weierstrass", (long) number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = deprecated__f_weierstrass_evaluate;
  problem->data = data;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }

  /* Calculate best parameter value */
  deprecated__f_weierstrass_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

#undef WEIERSTRASS_SUMMANDS
