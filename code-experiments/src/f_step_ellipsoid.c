/**
 * @file f_step_ellipsoid.c
 * @brief Implementation of the step ellipsoid function and problem.
 *
 * The BBOB step ellipsoid function intertwines the variable and objective transformations in such a way
 * that it is hard to devise a composition of generic transformations to implement it. In the end one would
 * have to implement several custom transformations which would be used solely by this problem. Therefore
 * we opt to implement it as a monolithic function instead.
 *
 * TODO: It would be nice to have a generic step ellipsoid function to complement this one.
 */
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "coco_utilities.c"
#include "suite_bbob_legacy_code.c"

/**
 * @brief Data type for the step ellipsoid problem.
 */
typedef struct {
  double *x, *xx;
  double *xopt, fopt;
  double **rot1, **rot2;
} f_step_ellipsoid_data_t;

/**
 * @brief Implements the step ellipsoid function without connections to any COCO structures.
 */
static double f_step_ellipsoid_raw(const double *x, const size_t number_of_variables, f_step_ellipsoid_data_t *data) {

  static const double condition = 100;
  static const double alpha = 10.0;
  size_t i, j;
  double penalty = 0.0, x1;
  double result;

  assert(number_of_variables > 1);

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

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
      data->x[i] = coco_double_round(data->x[i]);
    else
      data->x[i] = coco_double_round(alpha * data->x[i]) / alpha;
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
  result = 0.1 * coco_double_max(fabs(x1) * 1.0e-4, result) + penalty + data->fopt;

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_step_ellipsoid_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_step_ellipsoid_raw(x, problem->number_of_variables, (f_step_ellipsoid_data_t *) problem->data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the step ellipsoid data object.
 */
static void f_step_ellipsoid_free(coco_problem_t *problem) {
  f_step_ellipsoid_data_t *data;
  data = (f_step_ellipsoid_data_t *) problem->data;
  coco_free_memory(data->x);
  coco_free_memory(data->xx);
  coco_free_memory(data->xopt);
  bbob2009_free_matrix(data->rot1, problem->number_of_variables);
  bbob2009_free_matrix(data->rot2, problem->number_of_variables);
  /* Let the generic free problem code deal with all of the coco_problem_t fields */
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Creates the BBOB step ellipsoid problem.
 *
 * @note There is no separate basic allocate function.
 */
static coco_problem_t *f_step_ellipsoid_bbob_problem_allocate(const size_t function,
                                                              const size_t dimension,
                                                              const size_t instance,
                                                              const long rseed,
                                                              const char *problem_id_template,
                                                              const char *problem_name_template) {

  f_step_ellipsoid_data_t *data;
  size_t i;
  coco_problem_t *problem = coco_problem_allocate_from_scalars("step ellipsoid function",
      f_step_ellipsoid_evaluate, f_step_ellipsoid_free, dimension, -5.0, 5.0, 0);

  data = (f_step_ellipsoid_data_t *) coco_allocate_memory(sizeof(*data));
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
  for (i = 0; i < problem->number_of_variables; i++) {
      problem->best_parameter[i] = data->xopt[i];
  }
  problem->best_value[0] = data->fopt;

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "2-moderate");

  return problem;
}
