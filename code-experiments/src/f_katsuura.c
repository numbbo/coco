/**
 * @file f_katsuura.c
 * @brief Implementation of the Katsuura function and problem.
 */

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "coco_utilities.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_shift.c"
#include "transform_vars_affine.c"
#include "transform_vars_shift.c"
#include "transform_obj_penalize.c"

/**
 * @brief Implements the Katsuura function without connections to any COCO structures.
 */
static double f_katsuura_raw(const double *x, const size_t number_of_variables) {

  size_t i, j;
  double tmp, tmp2;
  double result;

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  /* Computation core */
  result = 1.0;
  for (i = 0; i < number_of_variables; ++i) {
    tmp = 0;
    for (j = 1; j < 33; ++j) {
      tmp2 = pow(2., (double) j);
      tmp += fabs(tmp2 * x[i] - coco_double_round(tmp2 * x[i])) / tmp2;
    }
    tmp = 1.0 + ((double) (long) i + 1) * tmp;
    result *= tmp;
  }
  result = 10. / ((double) number_of_variables) / ((double) number_of_variables)
      * (-1. + pow(result, 10. / pow((double) number_of_variables, 1.2)));

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_katsuura_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_katsuura_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Katsuura problem.
 */
static coco_problem_t *f_katsuura_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Katsuura function",
      f_katsuura_evaluate, NULL, number_of_variables, -5.0, 5.0, 1);
  coco_problem_set_id(problem, "%s_d%02lu", "katsuura", number_of_variables);

  /* Compute best solution */
  f_katsuura_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Katsuura problem.
 */
static coco_problem_t *f_katsuura_bbob_problem_allocate(const size_t function,
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

  const double penalty_factor = 1.0;

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
        double exponent = 1.0 * (int) k / ((double) (long) dimension - 1.0);
        current_row[j] += rot1[i][k] * pow(sqrt(100), exponent) * rot2[k][j];
      }
    }
  }

  problem = f_katsuura_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_penalize(problem, penalty_factor);

  bbob2009_free_matrix(rot1, dimension);
  bbob2009_free_matrix(rot2, dimension);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "5-weakly-structured");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}
