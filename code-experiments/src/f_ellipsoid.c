/**
 * @file f_ellipsoid.c
 * @brief Implementation of the ellipsoid function and problem.
 */

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "transform_vars_oscillate.c"
#include "transform_vars_affine.c"
#include "transform_vars_shift.c"
#include "transform_obj_shift.c"
#include "transform_vars_permutation.c"
#include "transform_vars_blockrotation.c"
#include "transform_obj_norm_by_dim.c"

/**
 * @brief Implements the ellipsoid function without connections to any COCO structures.
 */
static double f_ellipsoid_raw(const double *x, const size_t number_of_variables, double condition) {

  size_t i = 0;
  double result;
    
  if (coco_vector_contains_nan(x, number_of_variables))
    return NAN;

  result = x[i] * x[i];
  for (i = 1; i < number_of_variables; ++i) {
    const double exponent = 1.0 * (double) (long) i / ((double) (long) number_of_variables - 1.0);
    result += pow(condition, exponent) * x[i] * x[i];
  }

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_ellipsoid_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_ellipsoid_raw(x, problem->number_of_variables, problem -> condition);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Evaluates the gradient of the ellipsoid function.
 */
static void f_ellipsoid_evaluate_gradient(coco_problem_t *problem, 
                                          const double *x, 
                                          double *y) {

  double exponent;
  size_t i = 0;

  for (i = 0; i < problem->number_of_variables; ++i) {
    exponent = 1.0 * (double) (long) i / ((double) (long) problem->number_of_variables - 1.0);
    y[i] = 2.0*pow(problem -> condition, exponent) * x[i];
  }
 
}

/**
 * @brief Allocates the basic ellipsoid problem.
 */
static coco_problem_t *f_ellipsoid_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("ellipsoid function",
      f_ellipsoid_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  problem->evaluate_gradient = f_ellipsoid_evaluate_gradient;
  coco_problem_set_id(problem, "%s_d%02lu", "ellipsoid", number_of_variables);

  /* Compute best solution */
  f_ellipsoid_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB ellipsoid problem.
 */
static coco_problem_t *f_ellipsoid_bbob_problem_allocate(const size_t function,
                                                         const size_t dimension,
                                                         const size_t instance,
                                                         const long rseed,
                                                         const double conditioning,
                                                         const char *problem_id_template,
                                                         const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  problem = f_ellipsoid_allocate(dimension);
  problem -> condition = conditioning;
  problem = transform_vars_oscillate(problem);
  problem = transform_vars_shift(problem, xopt, 0);

  /*if large scale test-bed, normalize by dim*/
  if (coco_strfind(problem_name_template, "BBOB large-scale suite") >= 0){
    problem = transform_obj_norm_by_dim(problem);
  }
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

  coco_free_memory(xopt);
  return problem;
}

/**
 * @brief Creates the BBOB rotated ellipsoid problem.
 */
static coco_problem_t *f_ellipsoid_rotated_bbob_problem_allocate(const size_t function,
                                                                 const size_t dimension,
                                                                 const size_t instance,
                                                                 const long rseed,
                                                                 const double conditioning,
                                                                 const char *problem_id_template,
                                                                 const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  
  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double **rot1;

  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  fopt = bbob2009_compute_fopt(function, instance);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  bbob2009_free_matrix(rot1, dimension);

  problem = f_ellipsoid_allocate(dimension);
  problem -> condition = conditioning;
  problem = transform_vars_oscillate(problem);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "3-ill-conditioned");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}

/**
 * @brief Creates the BBOB permuted block-rotated ellipsoid problem.
 */
static coco_problem_t *f_ellipsoid_permblockdiag_bbob_problem_allocate(const size_t function,
                                                                       const size_t dimension,
                                                                       const size_t instance,
                                                                       const long rseed,
                                                                       const double conditioning,
                                                                       const char *problem_id_template,
                                                                       const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  double **B;
  const double *const *B_copy;
  size_t *P1 = coco_allocate_vector_size_t(dimension);
  size_t *P2 = coco_allocate_vector_size_t(dimension);
  size_t *block_sizes;
  size_t nb_blocks;
  size_t swap_range;
  size_t nb_swaps;
  
  block_sizes = coco_get_block_sizes(&nb_blocks, dimension, "bbob-largescale");
  swap_range = coco_get_swap_range(dimension, "bbob-largescale");
  nb_swaps = coco_get_nb_swaps(dimension, "bbob-largescale");
  /*printf("probDim:%d, scalingFactor: %f, blockSize: %d, swapRange: %d\n", dimension, scaling_factor, block_sizes[0], swap_range);*/

  
  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  
  B = coco_allocate_blockmatrix(dimension, block_sizes, nb_blocks);
  coco_compute_blockrotation(B, rseed + 1000000, dimension, block_sizes, nb_blocks);
  B_copy = (const double *const *)B;/*TODO: silences the warning, not sure if it prevents the modification of B at all levels. Check everywhere*/
  
  coco_compute_truncated_uniform_swap_permutation(P1, rseed + 2000000, dimension, nb_swaps, swap_range);
  coco_compute_truncated_uniform_swap_permutation(P2, rseed + 3000000, dimension, nb_swaps, swap_range);

  problem = f_ellipsoid_allocate(dimension);
  problem -> condition = conditioning;
  problem = transform_vars_oscillate(problem);
  problem = transform_vars_permutation(problem, P2, dimension);
  problem = transform_vars_blockrotation(problem, B_copy, dimension, block_sizes, nb_blocks);
  problem = transform_vars_permutation(problem, P1, dimension);
  problem = transform_vars_shift(problem, xopt, 0);

  
  problem = transform_obj_norm_by_dim(problem);
  problem = transform_obj_shift(problem, fopt);
  
  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "3-ill-conditioned");
  
  coco_free_block_matrix(B, dimension);
  coco_free_memory(P1);
  coco_free_memory(P2);
  coco_free_memory(block_sizes);
  coco_free_memory(xopt);
  return problem;
}

/**
 * @brief Creates the ellipsoid problem for the constrained BBOB suite
 */
static coco_problem_t *f_ellipsoid_cons_bbob_problem_allocate(const size_t function,
                                                         const size_t dimension,
                                                         const size_t instance,
                                                         const long rseed,
                                                         const double conditioning, 
                                                         const char *problem_id_template,
                                                         const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  problem = f_ellipsoid_allocate(dimension);
  problem -> condition = conditioning;
  /* TODO (NH): fopt -= problem->evaluate(all_zeros(dimension)) */
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);
  
  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

  coco_free_memory(xopt);
  return problem;
}

/**
 * @brief Creates the rotated ellipsoid problem for the constrained
 *        BBOB suite
 */
static coco_problem_t *f_ellipsoid_rotated_cons_bbob_problem_allocate(const size_t function,
                                                                 const size_t dimension,
                                                                 const size_t instance,
                                                                 const long rseed,
                                                                 const double conditioning,
                                                                 const char *problem_id_template,
                                                                 const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double **rot1;

  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  fopt = bbob2009_compute_fopt(function, instance);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  bbob2009_free_matrix(rot1, dimension);

  problem = f_ellipsoid_allocate(dimension);
  problem -> condition = conditioning;
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "3-ill-conditioned");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}
