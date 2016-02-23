/**
 * @file f_sharp_ridge.c
 * @brief Implementation of the sharp ridge function and problem.
 */

#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_shift.c"
#include "transform_vars_affine.c"
#include "transform_vars_shift.c"
#include "transform_vars_conditioning.c"

#include "transform_vars_permutation.c"
#include "transform_vars_blockrotation.c"
#include "transform_obj_norm_by_dim.c"

/**
 * @brief Implements the sharp ridge function without connections to any COCO structures.
 */
static double f_sharp_ridge_raw(const double *x, const size_t number_of_variables) {

  static const double alpha = 100.0;
  size_t i = 0;
  double result;

  assert(number_of_variables > 1);

  result = 0.0;
  for (i = 1; i < number_of_variables; ++i) {
    result += x[i] * x[i];
  }
  result = alpha * sqrt(result) + x[0] * x[0];

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_sharp_ridge_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_sharp_ridge_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic sharp ridge problem.
 */
static coco_problem_t *f_sharp_ridge_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("sharp ridge function",
      f_sharp_ridge_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "sharp_ridge", number_of_variables);

  /* Compute best solution */
  f_sharp_ridge_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB sharp ridge problem.
 */
static coco_problem_t *f_sharp_ridge_bbob_problem_allocate(const size_t function,
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
  bbob2009_free_matrix(rot1, dimension);
  bbob2009_free_matrix(rot2, dimension);
  problem = f_sharp_ridge_allocate(dimension);
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

/**
 * @brief Creates the BBOB permuted block-rotated sharp ridge problem
 */
static coco_problem_t *f_sharp_ridge_permblockdiag_bbob_problem_allocate(const size_t function,
                                                                         const size_t dimension,
                                                                         const size_t instance,
                                                                         const long rseed,
                                                                         const char *problem_id_template,
                                                                         const char *problem_name_template) {
    double *xopt, fopt;
    coco_problem_t *problem = NULL;
    double **B1;
    double **B2;
    const double *const *B1_copy;
    const double *const *B2_copy;
    size_t *P11 = coco_allocate_vector_size_t(dimension);
    size_t *P21 = coco_allocate_vector_size_t(dimension);
    size_t *P12 = coco_allocate_vector_size_t(dimension);
    size_t *P22 = coco_allocate_vector_size_t(dimension);
    size_t *block_sizes1;
    size_t *block_sizes2;
    size_t nb_blocks1;
    size_t nb_blocks2;
    size_t swap_range;
    size_t nb_swaps;
    
    block_sizes1 = coco_get_block_sizes(&nb_blocks1, dimension, "bbob-largescale");
    block_sizes2 = coco_get_block_sizes(&nb_blocks2, dimension, "bbob-largescale");
    swap_range = coco_get_swap_range(dimension, "bbob-largescale");
    nb_swaps = coco_get_nb_swaps(dimension, "bbob-largescale");
    
    xopt = coco_allocate_vector(dimension);
    fopt = bbob2009_compute_fopt(function, instance);
    bbob2009_compute_xopt(xopt, rseed, dimension);
    
    B1 = coco_allocate_blockmatrix(dimension, block_sizes1, nb_blocks1);
    B2 = coco_allocate_blockmatrix(dimension, block_sizes2, nb_blocks2);
    B1_copy = (const double *const *)B1;
    B2_copy = (const double *const *)B2;
    
    coco_compute_blockrotation(B1, rseed + 1000000, dimension, block_sizes1, nb_blocks1);
    coco_compute_blockrotation(B2, rseed + 2000000, dimension, block_sizes2, nb_blocks2);
    
    coco_compute_truncated_uniform_swap_permutation(P11, rseed + 3000000, dimension, nb_swaps, swap_range);
    coco_compute_truncated_uniform_swap_permutation(P21, rseed + 4000000, dimension, nb_swaps, swap_range);
    coco_compute_truncated_uniform_swap_permutation(P12, rseed + 5000000, dimension, nb_swaps, swap_range);
    coco_compute_truncated_uniform_swap_permutation(P22, rseed + 6000000, dimension, nb_swaps, swap_range);
    
    
    problem = f_sharp_ridge_allocate(dimension);
    problem = transform_vars_permutation(problem, P21, dimension);/* LIFO */
    problem = transform_vars_blockrotation(problem, B1_copy, dimension, block_sizes1, nb_blocks1);
    problem = transform_vars_permutation(problem, P11, dimension);
    problem = transform_vars_conditioning(problem, 10.0);
    problem = transform_vars_permutation(problem, P22, dimension);/*Consider replacing P11 and 22 by a single permutation P3*/
    problem = transform_vars_blockrotation(problem, B2_copy, dimension, block_sizes2, nb_blocks2);
    problem = transform_vars_permutation(problem, P12, dimension);
    problem = transform_vars_shift(problem, xopt, 0);
    
    problem = transform_obj_norm_by_dim(problem);
    problem = transform_obj_shift(problem, fopt);
    
    coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
    coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
    coco_problem_set_type(problem, "large_scale_block_rotated"); /*TODO: no large scale prefix*/
    
    coco_free_block_matrix(B1, dimension);
    coco_free_block_matrix(B2, dimension);
    coco_free_memory(P11);
    coco_free_memory(P21);
    coco_free_memory(P12);
    coco_free_memory(P22);
    coco_free_memory(block_sizes1);
    coco_free_memory(block_sizes2);
    coco_free_memory(xopt);
    
    return problem;
}


