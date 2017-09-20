/**
 * @file f_bent_cigar.c
 * @brief Implementation of the bent cigar function and problem.
 */

#include <stdio.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_shift.c"
#include "transform_vars_affine.c"
#include "transform_vars_asymmetric.c"
#include "transform_vars_shift.c"

/**
 * @brief Implements the bent cigar function without connections to any COCO structures.
 */
static double f_bent_cigar_raw(const double *x, const size_t number_of_variables) {

  static const double condition = 1.0e6;
  size_t i;
  double result;
  
  if (coco_vector_contains_nan(x, number_of_variables))
    return NAN;

  result = x[0] * x[0];
  for (i = 1; i < number_of_variables; ++i) {
    result += condition * x[i] * x[i];
  }
  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_bent_cigar_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_bent_cigar_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
  
}

/**
 * @brief Evaluates the gradient of the bent cigar function.
 */
static void f_bent_cigar_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {

  static const double condition = 1.0e6;
  size_t i;

  y[0] = 2.0 * x[0];
  for (i = 1; i < problem->number_of_variables; ++i)
    y[i] = 2.0 * condition * x[i];

}

/**
 * @brief Allocates the basic bent cigar problem.
 */
static coco_problem_t *f_bent_cigar_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("bent cigar function",
      f_bent_cigar_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  problem->evaluate_gradient = f_bent_cigar_evaluate_gradient;
  coco_problem_set_id(problem, "%s_d%02lu", "bent_cigar", number_of_variables);

  /* Compute best solution */
  f_bent_cigar_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB bent cigar problem.
 */
static coco_problem_t *f_bent_cigar_bbob_problem_allocate(const size_t function,
                                                          const size_t dimension,
                                                          const size_t instance,
                                                          const long rseed,
                                                          const char *problem_id_template,
                                                          const char *problem_name_template) {

  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double **rot1;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed + 1000000, dimension);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  bbob2009_free_matrix(rot1, dimension);

  problem = f_bent_cigar_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_asymmetric(problem, 0.5);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "3-ill-conditioned");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}

/**
 * @brief Creates the bent cigar problem for the constrained BBOB suite.
 */
static coco_problem_t *f_bent_cigar_cons_bbob_problem_allocate(const size_t function,
                                                          const size_t dimension,
                                                          const size_t instance,
                                                          const long rseed,
                                                          const char *problem_id_template,
                                                          const char *problem_name_template) {

  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double **rot1;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed + 1000000, dimension);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  bbob2009_free_matrix(rot1, dimension);

  problem = f_bent_cigar_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  
  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "3-ill-conditioned");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}
