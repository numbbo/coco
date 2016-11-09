/**
 * @file f_weierstrass.c
 * @brief Implementation of the Weierstrass function and problem.
 */

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_penalize.c"
#include "transform_obj_shift.c"
#include "transform_vars_affine.c"
#include "transform_vars_oscillate.c"
#include "transform_vars_shift.c"

#include "transform_vars_permutation.c"
#include "transform_vars_blockrotation.c"
#include "transform_obj_norm_by_dim.c"

/** @brief Number of summands in the Weierstrass problem. */
#define F_WEIERSTRASS_SUMMANDS 12

/**
 * @brief Data type for the Weierstrass problem.
 */
typedef struct {
  double f0;
  double ak[F_WEIERSTRASS_SUMMANDS];
  double bk[F_WEIERSTRASS_SUMMANDS];
} f_weierstrass_data_t;

/**
 * @brief Implements the Weierstrass function without connections to any COCO structures.
 */
static double f_weierstrass_raw(const double *x, const size_t number_of_variables, f_weierstrass_data_t *data) {

  size_t i, j;
  double result;

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    for (j = 0; j < F_WEIERSTRASS_SUMMANDS; ++j) {
      result += cos(2 * coco_pi * (x[i] + 0.5) * data->bk[j]) * data->ak[j];
    }
  }
  result = 10.0 * pow(result / (double) (long) number_of_variables - data->f0, 3.0);

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_weierstrass_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_weierstrass_raw(x, problem->number_of_variables, (f_weierstrass_data_t *) problem->data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Weierstrass problem.
 */
static coco_problem_t *f_weierstrass_allocate(const size_t number_of_variables) {

  f_weierstrass_data_t *data;
  size_t i;
  double *non_unique_best_value;
  coco_problem_t *problem = coco_problem_allocate_from_scalars("Weierstrass function",
      f_weierstrass_evaluate, NULL, number_of_variables, -5.0, 5.0, 0);
  coco_problem_set_id(problem, "%s_d%02lu", "weierstrass", number_of_variables);

  data = (f_weierstrass_data_t *) coco_allocate_memory(sizeof(*data));
  data->f0 = 0.0;
  for (i = 0; i < F_WEIERSTRASS_SUMMANDS; ++i) {
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

/**
 * @brief Creates the BBOB Weierstrass problem.
 */
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
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_oscillate(problem);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_penalize(problem, penalty_factor);

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
 * @brief Creates the BBOB permuted block-rotated Weierstrass problem.
 */
static coco_problem_t *f_weierstrass_permblockdiag_bbob_problem_allocate(const size_t function,
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
    const double condition = 100.0, penalty_factor = 10.0 / (double) dimension;
  
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
    
    problem = f_weierstrass_allocate(dimension);
    problem = transform_vars_permutation(problem, P22, dimension);
    problem = transform_vars_blockrotation(problem, B2_copy, dimension, block_sizes2, nb_blocks2);
    problem = transform_vars_permutation(problem, P12, dimension);
    
    problem = transform_vars_conditioning(problem, 1.0/condition);
    problem = transform_vars_permutation(problem, P21, dimension);
    problem = transform_vars_blockrotation(problem, B1_copy, dimension, block_sizes1, nb_blocks1);
    problem = transform_vars_permutation(problem, P11, dimension);
    
    problem = transform_vars_oscillate(problem);
    problem = transform_vars_permutation(problem, P22, dimension);
    problem = transform_vars_blockrotation(problem, B2_copy, dimension, block_sizes2, nb_blocks2);
    problem = transform_vars_permutation(problem, P12, dimension);

    problem = transform_vars_shift(problem, xopt, 0);
    /*problem = transform_obj_norm_by_dim(problem);*/ /* Wassim: there is already a normalization by dimension*/
    problem = transform_obj_penalize(problem, penalty_factor);
    problem = transform_obj_shift(problem, fopt);
    
    
    coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
    coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
    coco_problem_set_type(problem, "large_scale_block_rotated");
    
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

#undef F_WEIERSTRASS_SUMMANDS
