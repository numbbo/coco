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
#include "transform_obj_norm_by_dim.c"

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
    /*result *= tmp;*/ /* Wassim TODO: delete once consistency check passed*/
    result *= pow(tmp, 10. / pow((double) number_of_variables, 1.2));
  }
  /*result = 10. / ((double) number_of_variables) / ((double) number_of_variables)
      * (-1. + pow(result, 10. / pow((double) number_of_variables, 1.2)));*/
  result = 10. / (((double) number_of_variables) * ((double) number_of_variables))
  * (-1. + result);

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
      f_katsuura_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
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
  problem = transform_obj_shift(problem, fopt); /*There is no shift 'fopt' in the definition*/
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

/**
 * @brief Creates the BBOB permuted block-rotated Katsuura problem.
 */
static coco_problem_t *f_katsuura_permblockdiag_bbob_problem_allocate(const size_t function,
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
  const double penalty_factor = 1.0;
  size_t *P11 = coco_allocate_vector_size_t(dimension);
  size_t *P12 = coco_allocate_vector_size_t(dimension);
  size_t *P21 = coco_allocate_vector_size_t(dimension);
  size_t *P22 = coco_allocate_vector_size_t(dimension);
  size_t *block_sizes1;
  size_t *block_sizes2;
  size_t nb_blocks1;
  size_t nb_blocks2;
  size_t swap_range1;
  size_t swap_range2;
  size_t nb_swaps1;
  size_t nb_swaps2;

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
  coco_compute_blockrotation(B2, rseed, dimension, block_sizes2, nb_blocks2);

  coco_compute_truncated_uniform_swap_permutation(P11, rseed + 3000000, dimension, nb_swaps1, swap_range1);
  coco_compute_truncated_uniform_swap_permutation(P12, rseed + 4000000, dimension, nb_swaps1, swap_range1);
  coco_compute_truncated_uniform_swap_permutation(P21, rseed + 5000000, dimension, nb_swaps2, swap_range2);
  coco_compute_truncated_uniform_swap_permutation(P22, rseed + 6000000, dimension, nb_swaps2, swap_range2);

  problem = f_katsuura_allocate(dimension);
  problem = transform_vars_permutation(problem, P22, dimension);
  problem = transform_vars_blockrotation(problem, B1_copy, dimension, block_sizes2, nb_blocks2);
  problem = transform_vars_permutation(problem, P21, dimension);
  problem = transform_vars_conditioning(problem, 100.0);
  problem = transform_vars_permutation(problem, P12, dimension);
  problem = transform_vars_blockrotation(problem, B2_copy, dimension, block_sizes1, nb_blocks1);
  problem = transform_vars_permutation(problem, P11, dimension);
  problem = transform_vars_shift(problem, xopt, 0);

  /*problem = transform_obj_norm_by_dim(problem);*/ /* Wassim: does not seem to be needed*/
  problem = transform_obj_penalize(problem, penalty_factor);
  problem = transform_obj_shift(problem, fopt); /*TODO: documentation, there is no fopt in the definition of this function*/

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "5-weakly-structured");

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

