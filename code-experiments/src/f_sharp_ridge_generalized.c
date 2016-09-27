/**
 * @file f_sharp_ridge_generalized.c
 * @brief Implementation of the generalized sharp ridge function and problem.
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
 * @brief Data type for the versatile_data_t
 */
typedef struct {
  size_t proportion_of_linear_dims;
} f_sharp_ridge_generalized_versatile_data_t;


/**
 * @brief allows to free the versatile_data part of the problem.
 */
static void f_sharp_ridge_generalized_versatile_data_free(coco_problem_t *problem) {
  
  f_sharp_ridge_generalized_versatile_data_t *versatile_data = (f_sharp_ridge_generalized_versatile_data_t *) problem->versatile_data;
  coco_free_memory(versatile_data);
  problem->versatile_data = NULL;
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}


/**
 * @brief Implements the generalized sharp ridge function without connections to any COCO structures.
 */
static double f_sharp_ridge_generalized_raw(const double *x, const size_t number_of_variables, f_sharp_ridge_generalized_versatile_data_t* f_sharp_ridge_generalized_versatile_data) {

  static const double alpha = 100.0;
  size_t i = 0, number_linear_dimensions;
  double result;

  assert(number_of_variables > 1);

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  result = 0.0;
  number_linear_dimensions = number_of_variables / f_sharp_ridge_generalized_versatile_data->proportion_of_linear_dims;
  if (number_of_variables % f_sharp_ridge_generalized_versatile_data->proportion_of_linear_dims != 0) {
    number_linear_dimensions += 1;
  }
  for (i = number_linear_dimensions; i < number_of_variables; ++i) {
    result += x[i] * x[i];
  }
  result = alpha * sqrt(result);
  for (i = 0; i < number_linear_dimensions; i++) {
    result += x[i] * x[i];
  }

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_sharp_ridge_generalized_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_sharp_ridge_generalized_raw(x, problem->number_of_variables, (f_sharp_ridge_generalized_versatile_data_t *)problem->versatile_data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic sharp ridge problem.
 */
static coco_problem_t *f_sharp_ridge_generalized_allocate(const size_t number_of_variables, size_t proportion_of_linear_dims) {
  /* Wassim: proportion_of_linear_dims should probably be allowed to be non-integer */
  coco_problem_t *problem = coco_problem_allocate_from_scalars("sharp ridge function",
      f_sharp_ridge_generalized_evaluate, f_sharp_ridge_generalized_versatile_data_free, number_of_variables, -5.0, 5.0, 0.0);

  coco_problem_set_id(problem, "%s_d%02lu", "sharp_ridge_generalized", number_of_variables);
  problem->versatile_data = (f_sharp_ridge_generalized_versatile_data_t *) coco_allocate_memory(sizeof(f_sharp_ridge_generalized_versatile_data_t));
  ((f_sharp_ridge_generalized_versatile_data_t *) problem->versatile_data)->proportion_of_linear_dims = proportion_of_linear_dims;

  /* Compute best solution */
  f_sharp_ridge_generalized_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}


/**
 * @brief Creates the BBOB permuted block-rotated generalized sharp ridge problem
 */
static coco_problem_t *f_sharp_ridge_generalized_permblockdiag_bbob_problem_allocate(const size_t function,
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

  const size_t proportion_of_linear_dims = 40;

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
  
  
  problem = f_sharp_ridge_generalized_allocate(dimension, proportion_of_linear_dims);
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


