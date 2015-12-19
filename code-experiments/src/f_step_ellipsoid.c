/*
 * f_step_ellipsoid.c
 *
 * The BBOB step ellipsoid function intertwines the variable and objective transformations in such a way
 * that it is hard to devise a composition of generic transformations to implement it. In the end one would
 * have to implement several custom transformations which would be used solely by this problem. Therefore
 * we opt to implement it as a monolithic function instead.
 *
 * TODO: It would be nice to have a generic step ellipsoid function to complement this one.
 */
#include "coco_platform.h"

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "coco_utilities.c"
#include "suite_bbob2009_legacy_code.c"

typedef struct {
  double *x, *xx;
  double *xopt, fopt;
  double **rot1, **rot2;
} f_step_ellipsoid_data_t;

static double f_step_ellipsoid_raw(const double *x, size_t number_of_variables, f_step_ellipsoid_data_t *data) {

  static const double condition = 100;
  static const double alpha = 10.0;
  size_t i, j;
  double penalty = 0.0, x1;
  double result;

  assert(number_of_variables > 1);

  for (i = 0; i < number_of_variables; ++i) {
    double tmp;
    tmp = fabs(x[i]) - 5.0;
    if (tmp > 0.0)
      penalty += tmp * tmp;
  }

  for (i = 0; i < number_of_variables; ++i) {
    double c1;
    data->x[i] = 0.0;
    c1 = sqrt(pow(condition / 10., (double) i / (double) (number_of_variables - 1)));
    for (j = 0; j < number_of_variables; ++j) {
      data->x[i] += c1 * data->rot2[i][j] * (x[j] - data->xopt[j]);
    }
  }
  x1 = data->x[0];

  for (i = 0; i < number_of_variables; ++i) {
    if (fabs(data->x[i]) > 0.5)
      data->x[i] = coco_round_double(data->x[i]);
    else
      data->x[i] = coco_round_double(alpha * data->x[i]) / alpha;
  }

  for (i = 0; i < number_of_variables; ++i) {
    data->xx[i] = 0.0;
    for (j = 0; j < number_of_variables; ++j) {
      data->xx[i] += data->rot1[i][j] * data->x[j];
    }
  }

  /* Computation core */
  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    double exponent;
    exponent = (double) (long) i / ((double) (long) number_of_variables - 1.0);
    result += pow(condition, exponent) * data->xx[i] * data->xx[i];
    ;
  }
  result = 0.1 * coco_max_double(fabs(x1) * 1.0e-4, result) + penalty + data->fopt;

  return result;
}

static void f_step_ellipsoid_evaluate(coco_problem_t *self, const double *x, double *y) {
  assert(self->number_of_objectives == 1);
  y[0] = f_step_ellipsoid_raw(x, self->number_of_variables, self->data);
}

static void f_step_ellipsoid_free(coco_problem_t *self) {
  f_step_ellipsoid_data_t *data;
  data = self->data;
  coco_free_memory(data->x);
  coco_free_memory(data->xx);
  coco_free_memory(data->xopt);
  bbob2009_free_matrix(data->rot1, self->number_of_variables);
  bbob2009_free_matrix(data->rot2, self->number_of_variables);
  /* Let the generic free problem code deal with all of the coco_problem_t fields */
  self->free_problem = NULL;
  coco_problem_free(self);
}

/* Note: there is no separate f_step_ellipsoid_allocate() function! */

static coco_problem_t *f_step_ellipsoid_bbob_problem_allocate(const size_t function,
                                                              const size_t dimension,
                                                              const size_t instance,
                                                              const long rseed,
                                                              const char *problem_id_template,
                                                              const char *problem_name_template) {

  f_step_ellipsoid_data_t *data;
  coco_problem_t *problem = coco_problem_allocate_from_scalars("step ellipsoid function",
      f_step_ellipsoid_evaluate, f_step_ellipsoid_free, dimension, -5.0, 5.0, NAN);

  data = coco_allocate_memory(sizeof(*data));
  /* Allocate temporary storage and space for the rotation matrices */
  data->x = coco_allocate_vector(dimension);
  data->xx = coco_allocate_vector(dimension);
  data->xopt = coco_allocate_vector(dimension);
  data->rot1 = bbob2009_allocate_matrix(dimension, dimension);
  data->rot2 = bbob2009_allocate_matrix(dimension, dimension);

  data->fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(data->xopt, rseed, dimension);
  bbob2009_compute_rotation(data->rot1, rseed + 1000000, dimension);
  bbob2009_compute_rotation(data->rot2, rseed, dimension);

  problem->data = data;

  /* Compute best solution
   *
   * OME: Dirty hack for now because I did not want to invert the
   * transformations to find the best_parameter :/
   */
  problem->best_value[0] = data->fopt;

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "2-moderate");

  return problem;
}

/* TODO: Deprecated functions below are to be deleted when the new ones work as they should */

static void deprecated__f_bbob_step_ellipsoid_evaluate(coco_problem_t *self, const double *x, double *y) {
  static const double condition = 100;
  static const double alpha = 10.0;
  size_t i, j;
  double penalty = 0.0, x1;
  f_step_ellipsoid_data_t *data;

  assert(self->number_of_variables > 1);
  assert(self->number_of_objectives == 1);

  data = self->data;
  for (i = 0; i < self->number_of_variables; ++i) {
    double tmp;
    tmp = fabs(x[i]) - 5.0;
    if (tmp > 0.0)
      penalty += tmp * tmp;
  }

  for (i = 0; i < self->number_of_variables; ++i) {
    double c1;
    data->x[i] = 0.0;
    c1 = sqrt(pow(condition / 10., (double) i / (double) (self->number_of_variables - 1)));
    for (j = 0; j < self->number_of_variables; ++j) {
      data->x[i] += c1 * data->rot2[i][j] * (x[j] - data->xopt[j]);
    }
  }
  x1 = data->x[0];

  for (i = 0; i < self->number_of_variables; ++i) {
    if (fabs(data->x[i]) > 0.5)
      data->x[i] = coco_round_double(data->x[i]);
    else
      data->x[i] = coco_round_double(alpha * data->x[i]) / alpha;
  }

  for (i = 0; i < self->number_of_variables; ++i) {
    data->xx[i] = 0.0;
    for (j = 0; j < self->number_of_variables; ++j) {
      data->xx[i] += data->rot1[i][j] * data->x[j];
    }
  }

  /* Computation core */
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    double exponent;
    exponent = (double) (long) i / ((double) (long) self->number_of_variables - 1.0);
    y[0] += pow(condition, exponent) * data->xx[i] * data->xx[i];
    ;
  }
  y[0] = 0.1 * coco_max_double(fabs(x1) * 1.0e-4, y[0]) + penalty + data->fopt;
}

static coco_problem_t *deprecated__f_bbob_step_ellipsoid(const size_t number_of_variables,
                                                         const size_t instance_id) {
  size_t i, problem_id_length;
  long rseed;
  coco_problem_t *problem;
  f_step_ellipsoid_data_t *data;

  rseed = (long) (7 + 10000 * instance_id);

  data = coco_allocate_memory(sizeof(*data));
  /* Allocate temporary storage and space for the rotation matrices */
  data->x = coco_allocate_vector(number_of_variables);
  data->xx = coco_allocate_vector(number_of_variables);
  data->xopt = coco_allocate_vector(number_of_variables);
  data->rot1 = bbob2009_allocate_matrix(number_of_variables, number_of_variables);
  data->rot2 = bbob2009_allocate_matrix(number_of_variables, number_of_variables);

  data->fopt = bbob2009_compute_fopt(7, instance_id);
  bbob2009_compute_xopt(data->xopt, rseed, number_of_variables);
  bbob2009_compute_rotation(data->rot1, rseed + 1000000, number_of_variables);
  bbob2009_compute_rotation(data->rot2, rseed, number_of_variables);

  problem = coco_problem_allocate(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("BBOB f7");
  /* Construct a meaningful problem id */
  problem_id_length = (size_t) snprintf(NULL, 0, "%s_%02lu", "bbob2009_f7", (long) number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02lu", "bbob2009_f7", (long) number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->data = data;
  problem->evaluate_function = deprecated__f_bbob_step_ellipsoid_evaluate;
  problem->free_problem = f_step_ellipsoid_free;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = NAN;
  }
  /* "Calculate" best parameter value.
   *
   * OME: Dirty hack for now because I did not want to invert the
   * transformations to find the best_parameter :/
   */
  problem->best_value[0] = data->fopt;
  return problem;
}
