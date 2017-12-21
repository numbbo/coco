/**
 * @file f_bent_cigar_generalized.c
 * @brief Implementation of the generalized bent cigar function and problem.
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

#include "transform_vars_permutation.c"
#include "transform_vars_blockrotation.c"
#include "transform_obj_norm_by_dim.c"

/**
 * @brief Data type for the versatile_data_t
 */
typedef struct {
  size_t proportion_long_axes_denom;
} f_bent_cigar_generalized_versatile_data_t;

/**
 * @brief allows to free the versatile_data part of the problem.
 */
static void f_bent_cigar_generalized_versatile_data_free(coco_problem_t *problem) {

  f_bent_cigar_generalized_versatile_data_t *versatile_data = (f_bent_cigar_generalized_versatile_data_t *) problem->versatile_data;
  coco_free_memory(versatile_data);
  problem->versatile_data = NULL;
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Implements the generalized bent cigar function without connections to any COCO structures.
 */
static double f_bent_cigar_generalized_raw(const double *x, const size_t number_of_variables, f_bent_cigar_generalized_versatile_data_t* f_bent_cigar_generalized_versatile_data) {

  static const double condition = 1.0e6;
  size_t i, nb_long_axes;
  double result;
  result = 0;
  nb_long_axes = number_of_variables / f_bent_cigar_generalized_versatile_data->proportion_long_axes_denom;
  if (number_of_variables % f_bent_cigar_generalized_versatile_data->proportion_long_axes_denom != 0) {
    nb_long_axes += 1;
  }
  for (i = 0; i < nb_long_axes; ++i) {
    result += x[i] * x[i];
  }
  for (i = nb_long_axes; i < number_of_variables; ++i) {
    result += condition * x[i] * x[i];
  }
  return result;
}

/**
 * @brief Uses the generalized raw function to evaluate the COCO problem.
 */
static void f_bent_cigar_generalized_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_bent_cigar_generalized_raw(x, problem->number_of_variables, (f_bent_cigar_generalized_versatile_data_t *)problem->versatile_data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic generalized bent cigar problem.
 */
static coco_problem_t *f_bent_cigar_generalized_allocate(const size_t number_of_variables, size_t proportion_long_axes_denom) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("generalized bent cigar function",
                                                               f_bent_cigar_generalized_evaluate, f_bent_cigar_generalized_versatile_data_free, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%04lu", "bent_cigar", number_of_variables);
  problem->versatile_data = (f_bent_cigar_generalized_versatile_data_t *) coco_allocate_memory(sizeof(f_bent_cigar_generalized_versatile_data_t));
  ((f_bent_cigar_generalized_versatile_data_t *) problem->versatile_data)->proportion_long_axes_denom = proportion_long_axes_denom;

  /* Compute best solution */
  f_bent_cigar_generalized_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}


/**
 * @brief Creates the BBOB generalized permuted block-rotated bent cigar problem.
 */
static coco_problem_t *f_bent_cigar_generalized_permblockdiag_bbob_problem_allocate(const size_t function,
                                                                                    const size_t dimension,
                                                                                    const size_t instance,
                                                                                    const long rseed,
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
  const size_t proportion_long_axes_denom = 40;

  block_sizes = coco_get_block_sizes(&nb_blocks, dimension, "bbob-largescale");
  swap_range = coco_get_swap_range(dimension, "bbob-largescale");
  nb_swaps = coco_get_nb_swaps(dimension, "bbob-largescale");

  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed + 1000000, dimension);
  fopt = bbob2009_compute_fopt(function, instance);

  B = coco_allocate_blockmatrix(dimension, block_sizes, nb_blocks);
  B_copy = (const double *const *)B;

  coco_compute_blockrotation(B, rseed + 1000000, dimension, block_sizes, nb_blocks);
  coco_compute_truncated_uniform_swap_permutation(P1, rseed + 2000000, dimension, nb_swaps, swap_range);
  coco_compute_truncated_uniform_swap_permutation(P2, rseed + 3000000, dimension, nb_swaps, swap_range);

  problem = f_bent_cigar_generalized_allocate(dimension, proportion_long_axes_denom);
  problem = transform_vars_permutation(problem, P2, dimension);
  problem = transform_vars_blockrotation(problem, B_copy, dimension, block_sizes, nb_blocks);
  problem = transform_vars_permutation(problem, P1, dimension);
  problem = transform_vars_asymmetric(problem, 0.5);
  problem = transform_vars_permutation(problem, P2, dimension);
  problem = transform_vars_blockrotation(problem, B_copy, dimension, block_sizes, nb_blocks);
  problem = transform_vars_permutation(problem, P1, dimension);
  problem = transform_vars_shift(problem, xopt, 0);

  problem = transform_obj_norm_by_dim(problem);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "block_rotated_ill-conditioned");
  coco_free_block_matrix(B, dimension);
  coco_free_memory(P1);
  coco_free_memory(P2);
  coco_free_memory(block_sizes);
  coco_free_memory(xopt);
  return problem;
}


