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

#include "transform_vars_permutation.c"
#include "transform_vars_blockrotation.c"
#include "transform_obj_norm_by_dim.c"

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
  if (coco_is_inf(sum2)) /* cos(inf) -> nan */
    return sum2;
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
 * @brief Creates the BBOB rotated Rastrigin problem for large scale.
 */
static coco_problem_t *f_rastrigin_permblockdiag_bbob_problem_allocate(const size_t function,
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
  size_t *P12 = coco_allocate_vector_size_t(dimension);
  size_t *P21 = coco_allocate_vector_size_t(dimension);
  size_t *P22 = coco_allocate_vector_size_t(dimension);
  size_t *block_sizes1, *block_sizes2;
  size_t nb_blocks1, nb_blocks2;
  size_t swap_range1, swap_range2;
  size_t nb_swaps1, nb_swaps2;

  block_sizes1 = coco_get_block_sizes(&nb_blocks1, dimension, "bbob-largescale");
  block_sizes2 = coco_get_block_sizes(&nb_blocks2, dimension, "bbob-largescale");
  swap_range1 = coco_get_swap_range(dimension, "bbob-largescale");
  swap_range2 = coco_get_swap_range(dimension, "bbob-largescale");
  nb_swaps1 = coco_get_nb_swaps(dimension, "bbob-largescale");
  nb_swaps2 = coco_get_nb_swaps(dimension, "bbob-largescale");

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  B1 = coco_allocate_blockmatrix(dimension, block_sizes1, nb_blocks1);
  B2 = coco_allocate_blockmatrix(dimension, block_sizes2, nb_blocks2);
  B1_copy = (const double *const *)B1;
  B2_copy = (const double *const *)B2;

  coco_compute_blockrotation(B1, rseed + 1000000, dimension, block_sizes1, nb_blocks1);
  coco_compute_blockrotation(B2, rseed + 2000000, dimension, block_sizes2, nb_blocks2);

  coco_compute_truncated_uniform_swap_permutation(P11, rseed + 3000000, dimension, nb_swaps1, swap_range1);
  coco_compute_truncated_uniform_swap_permutation(P12, rseed + 4000000, dimension, nb_swaps1, swap_range1);
  coco_compute_truncated_uniform_swap_permutation(P21, rseed + 5000000, dimension, nb_swaps2, swap_range2);
  coco_compute_truncated_uniform_swap_permutation(P22, rseed + 6000000, dimension, nb_swaps2, swap_range2);
  
  problem = f_rastrigin_allocate(dimension);
  problem = transform_vars_permutation(problem, P12, dimension);
  problem = transform_vars_blockrotation(problem, B1_copy, dimension, block_sizes1, nb_blocks1);
  problem = transform_vars_permutation(problem, P11, dimension);
  problem = transform_vars_conditioning(problem, 10.0);
  problem = transform_vars_permutation(problem, P22, dimension);
  problem = transform_vars_blockrotation(problem, B2_copy, dimension, block_sizes2, nb_blocks2);
  problem = transform_vars_permutation(problem, P21, dimension);
  problem = transform_vars_asymmetric(problem, 0.2);
  problem = transform_vars_oscillate(problem);
  problem = transform_vars_permutation(problem, P12, dimension);
  problem = transform_vars_blockrotation(problem, B1_copy, dimension, block_sizes1, nb_blocks1);
  problem = transform_vars_permutation(problem, P11, dimension);
  problem = transform_vars_shift(problem, xopt, 0);

  problem = transform_obj_norm_by_dim(problem);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "block-rotated_multi-modal");

  coco_free_block_matrix(B1, dimension);
  coco_free_block_matrix(B2, dimension);
  coco_free_memory(P11);
  coco_free_memory(P12);
  coco_free_memory(P21);
  coco_free_memory(P22);
  coco_free_memory(block_sizes1);
  coco_free_memory(block_sizes2);
  coco_free_memory(xopt);
  return problem;
}
