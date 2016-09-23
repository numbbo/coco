/**
 * @file f_rastrigin.c
 * @brief Implementation of the Rastrigin function and problem.
 */

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_vars_conditioning.c"
#include "transform_vars_asymmetric.c"
#include "transform_vars_oscillate.c"
#include "transform_vars_shift.c"
#include "transform_obj_shift.c"
#include "transform_vars_affine.c"

/**
 * @brief Implements the Rastrigin function without connections to any COCO structures.
 */
static double f_rastrigin_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double result;
  double sum1 = 0.0, sum2 = 0.0;
    
  if (coco_vector_contains_nan(x, number_of_variables))
    return NAN;

  for (i = 0; i < number_of_variables; ++i) {
    sum1 += cos(coco_two_pi * x[i]);
    sum2 += x[i] * x[i];
  }
  result = 10.0 * ((double) (long) number_of_variables - sum1) + sum2;

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_rastrigin_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_rastrigin_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Rastrigin problem.
 */
static coco_problem_t *f_rastrigin_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Rastrigin function",
      f_rastrigin_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "rastrigin", number_of_variables);

  /* Compute best solution */
  f_rastrigin_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Rastrigin problem.
 */
static coco_problem_t *f_rastrigin_bbob_problem_allocate(const size_t function,
                                                         const size_t dimension,
                                                         const size_t instance,
                                                         const long rseed,
                                                         const char *problem_id_template,
                                                         const char *problem_name_template) {

  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  problem = f_rastrigin_allocate(dimension);
  problem = transform_vars_conditioning(problem, 10.0);
  problem = transform_vars_asymmetric(problem, 0.2);
  problem = transform_vars_oscillate(problem);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

  coco_free_memory(xopt);
  return problem;
}

/**
 * @brief Creates the BBOB rotated Rastrigin problem.
 */
static coco_problem_t *f_rastrigin_rotated_bbob_problem_allocate(const size_t function,
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
        current_row[j] += rot1[i][k] * pow(sqrt(10), exponent) * rot2[k][j];
      }
    }
  }

  problem = f_rastrigin_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_asymmetric(problem, 0.2);
  problem = transform_vars_oscillate(problem);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);

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

/**
 * @brief Creates the Rastrigin problem for the constrained BBOB suite.
 */
static coco_problem_t *f_rastrigin_cons_bbob_problem_allocate(const size_t function,
                                                         const size_t dimension,
                                                         const size_t instance,
                                                         const long rseed,
                                                         const char *problem_id_template,
                                                         const char *problem_name_template) {

  double *xshift, fopt;
  coco_problem_t *problem = NULL;
  size_t i;

  xshift = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  
  for (i = 0; i < dimension; ++i)
    xshift[i] = -1.0;

  problem = f_rastrigin_allocate(dimension);
  problem = transform_vars_shift(problem, xshift, 0);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

  return problem;
}

