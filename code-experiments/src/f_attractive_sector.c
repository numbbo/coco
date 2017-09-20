/**
 * @file f_attractive_sector.c
 * @brief Implementation of the attractive sector function and problem.
 */

#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_oscillate.c"
#include "transform_obj_power.c"
#include "transform_obj_shift.c"
#include "transform_vars_affine.c"
#include "transform_vars_shift.c"

/**
 * @brief Data type for the attractive sector problem.
 */
typedef struct {
  double *xopt;
} f_attractive_sector_data_t;

/**
 * @brief Implements the attractive sector function without connections to any COCO structures.
 */
static double f_attractive_sector_raw(const double *x,
                                      const size_t number_of_variables,
                                      f_attractive_sector_data_t *data) {
  size_t i;
  double result;

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    if (data->xopt[i] * x[i] > 0.0) {
      result += 100.0 * 100.0 * x[i] * x[i];
    } else {
      result += x[i] * x[i];
    }
  }
  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_attractive_sector_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_attractive_sector_raw(x, problem->number_of_variables, (f_attractive_sector_data_t *) problem->data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the attractive sector data object.
 */
static void f_attractive_sector_free(coco_problem_t *problem) {
  f_attractive_sector_data_t *data;
  data = (f_attractive_sector_data_t *) problem->data;
  coco_free_memory(data->xopt);
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Allocates the basic attractive sector problem.
 */
static coco_problem_t *f_attractive_sector_allocate(const size_t number_of_variables, const double *xopt) {

  f_attractive_sector_data_t *data;
  coco_problem_t *problem = coco_problem_allocate_from_scalars("attractive sector function",
      f_attractive_sector_evaluate, f_attractive_sector_free, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "attractive_sector", number_of_variables);

  data = (f_attractive_sector_data_t *) coco_allocate_memory(sizeof(*data));
  data->xopt = coco_duplicate_vector(xopt, number_of_variables);
  problem->data = data;

  /* Compute best solution */
  f_attractive_sector_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB attractive sector problem.
 */
static coco_problem_t *f_attractive_sector_bbob_problem_allocate(const size_t function,
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

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  /* Compute affine transformation M from two rotation matrices */
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
        double exponent = 1.0 * (int) k / ((double) (long) dimension - 1.0);
        current_row[j] += rot1[i][k] * pow(sqrt(10.0), exponent) * rot2[k][j];
      }
    }
  }
  bbob2009_free_matrix(rot1, dimension);
  bbob2009_free_matrix(rot2, dimension);

  problem = f_attractive_sector_allocate(dimension, xopt);
  problem = transform_obj_oscillate(problem);
  problem = transform_obj_power(problem, 0.9);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "2-moderate");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}
