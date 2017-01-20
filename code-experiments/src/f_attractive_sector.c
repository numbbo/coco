/**
 * @file f_attractive_sector.c
 * @brief Implementation of the attractive sector function and problem.
 */

#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_oscillate.c"
#include "transform_obj_power.c"
#include "transform_obj_shift.c"
#include "transform_vars_affine.c"
#include "transform_vars_shift.c"

#include "transform_vars_permutation.c"
#include "transform_vars_blockrotation.c"
#include "transform_vars_conditioning.c"
#include "transform_obj_norm_by_dim.c"

/**
 * @brief Data type for the attractive sector problem.
 */
typedef struct {
  double *xopt;
} f_attractive_sector_data_t;

/**
 * @brief Implements the attractive sector function without connections to any COCO structures.
 */
static double f_attractive_sector_raw(const double *x,
                                      const size_t number_of_variables,
                                      f_attractive_sector_data_t *data) {
  size_t i;
  double result;

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    if (data->xopt[i] * x[i] > 0.0) {
      result += 100.0 * 100.0 * x[i] * x[i];
    } else {
      result += x[i] * x[i];
    }
  }
  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_attractive_sector_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_attractive_sector_raw(x, problem->number_of_variables, (f_attractive_sector_data_t *) problem->data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the attractive sector data object.
 */
static void f_attractive_sector_free(coco_problem_t *problem) {
  f_attractive_sector_data_t *data;
  data = (f_attractive_sector_data_t *) problem->data;
  coco_free_memory(data->xopt);
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Allocates the basic attractive sector problem.
 */
static coco_problem_t *f_attractive_sector_allocate(const size_t number_of_variables, const double *xopt) {

  f_attractive_sector_data_t *data;
  coco_problem_t *problem = coco_problem_allocate_from_scalars("attractive sector function",
      f_attractive_sector_evaluate, f_attractive_sector_free, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "attractive_sector", number_of_variables);

  data = (f_attractive_sector_data_t *) coco_allocate_memory(sizeof(*data));
  data->xopt = coco_duplicate_vector(xopt, number_of_variables);
  problem->data = data;

  /* Compute best solution */
  f_attractive_sector_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB attractive sector problem.
 */
static coco_problem_t *f_attractive_sector_bbob_problem_allocate(const size_t function,
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

  /* Compute affine transformation M from two rotation matrices */
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
        current_row[j] += rot1[i][k] * pow(sqrt(10.0), exponent) * rot2[k][j];
      }
    }
  }
  bbob2009_free_matrix(rot1, dimension);
  bbob2009_free_matrix(rot2, dimension);

  problem = f_attractive_sector_allocate(dimension, xopt);
  problem = transform_obj_oscillate(problem);
  problem = transform_obj_power(problem, 0.9);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "2-moderate");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}


static coco_problem_t *f_attractive_sector_permblockdiag_bbob_problem_allocate(const size_t function,
                                                                       const size_t dimension,
                                                                       const size_t instance,
                                                                       const long rseed,
                                                                       const char *problem_id_template,
                                                                       const char *problem_name_template) {
  
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  double **B1, **B2;
  const double *const *B1_copy;
  const double *const *B2_copy;
  const double condition = 10.0;
  size_t *P11, *P12, *P21, *P22;
  size_t *block_sizes1, *block_sizes2;/* each of R and Q migh have its own parameter values */
  size_t nb_blocks1, nb_blocks2;
  size_t swap_range1, swap_range2;
  size_t nb_swaps1, nb_swaps2;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  block_sizes1 = coco_get_block_sizes(&nb_blocks1, dimension, "bbob-largescale");
  block_sizes2 = coco_get_block_sizes(&nb_blocks2, dimension, "bbob-largescale");
  swap_range1 = coco_get_swap_range(dimension, "bbob-largescale");
  swap_range2 = coco_get_swap_range(dimension, "bbob-largescale");
  nb_swaps1 = coco_get_nb_swaps(dimension, "bbob-largescale");
  nb_swaps2 = coco_get_nb_swaps(dimension, "bbob-largescale");

  B1 = coco_allocate_blockmatrix(dimension, block_sizes1, nb_blocks1);
  B2 = coco_allocate_blockmatrix(dimension, block_sizes2, nb_blocks2);
  B1_copy = (const double *const *)B1;/* Wassim: silences the warning, not sure it prevents the modification of B at all levels*/
  B2_copy = (const double *const *)B2;
  coco_compute_blockrotation(B1, rseed + 1000000, dimension, block_sizes1, nb_blocks1);
  coco_compute_blockrotation(B2, rseed, dimension, block_sizes2, nb_blocks2);

  P11 = coco_allocate_vector_size_t(dimension);
  P12 = coco_allocate_vector_size_t(dimension);
  P21 = coco_allocate_vector_size_t(dimension);
  P22 = coco_allocate_vector_size_t(dimension);
  coco_compute_truncated_uniform_swap_permutation(P11, rseed + 2000000, dimension, nb_swaps1, swap_range1);
  coco_compute_truncated_uniform_swap_permutation(P12, rseed + 3000000, dimension, nb_swaps1, swap_range1);
  coco_compute_truncated_uniform_swap_permutation(P21, rseed + 4000000, dimension, nb_swaps2, swap_range2);
  coco_compute_truncated_uniform_swap_permutation(P22, rseed + 5000000, dimension, nb_swaps2, swap_range2);

  problem = f_attractive_sector_allocate(dimension, xopt);
  problem = transform_obj_norm_by_dim(problem);
  problem = transform_obj_oscillate(problem);
  problem = transform_obj_power(problem, 0.9);
  problem = transform_obj_shift(problem, fopt);

  problem = transform_vars_permutation(problem, P22, dimension);
  problem = transform_vars_blockrotation(problem, B2_copy, dimension, block_sizes2, nb_blocks2);
  problem = transform_vars_permutation(problem, P21, dimension);
  problem = transform_vars_conditioning(problem, condition);
  problem = transform_vars_permutation(problem, P12, dimension);
  problem = transform_vars_blockrotation(problem, B1_copy, dimension, block_sizes1, nb_blocks1);
  problem = transform_vars_permutation(problem, P11, dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  
  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "block_rotated_moderate");
  
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
