/**
 * @file f_different_powers.c
 * @brief Implementation of the different powers function and problem.
 */

#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_shift.c"
#include "transform_vars_affine.c"
#include "transform_vars_shift.c"

#include "transform_vars_permutation.c"
#include "transform_vars_blockrotation.c"
#include "transform_obj_norm_by_dim.c"

/**
 * @brief Implements the different powers function without connections to any COCO structures.
 */
static double f_different_powers_raw(const double *x, const size_t number_of_variables) {

  size_t i;
  double sum = 0.0;
  double result;

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  for (i = 0; i < number_of_variables; ++i) {
    double exponent = 2.0 + (4.0 * (double) (long) i) / ((double) (long) number_of_variables - 1.0);
    sum += pow(fabs(x[i]), exponent);
  }
  result = sqrt(sum);

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_different_powers_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_different_powers_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic different powers problem.
 */
static coco_problem_t *f_different_powers_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("different powers function",
      f_different_powers_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "different_powers", number_of_variables);

  /* Compute best solution */
  f_different_powers_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB different powers problem.
 */
static coco_problem_t *f_different_powers_bbob_problem_allocate(const size_t function,
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
  bbob2009_compute_xopt(xopt, rseed, dimension);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  bbob2009_free_matrix(rot1, dimension);

  problem = f_different_powers_allocate(dimension);
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
 * @brief Creates the BBOB generalized permuted block-rotated sum of different powers problem.
 */
static coco_problem_t *f_different_powers_permblockdiag_bbob_problem_allocate(const size_t function,
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

  block_sizes = coco_get_block_sizes(&nb_blocks, dimension, "bbob-largescale");
  swap_range = coco_get_swap_range(dimension, "bbob-largescale");
  nb_swaps = coco_get_nb_swaps(dimension, "bbob-largescale");

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  B = coco_allocate_blockmatrix(dimension, block_sizes, nb_blocks);
  B_copy = (const double *const *)B;
  coco_compute_blockrotation(B, rseed + 1000000, dimension, block_sizes, nb_blocks);

  coco_compute_truncated_uniform_swap_permutation(P1, rseed + 2000000, dimension, nb_swaps, swap_range);
  coco_compute_truncated_uniform_swap_permutation(P2, rseed + 3000000, dimension, nb_swaps, swap_range);

  problem = f_different_powers_allocate(dimension);
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
