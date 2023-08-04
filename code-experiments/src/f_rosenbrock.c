/**
 * @file f_rosenbrock.c
 * @brief Implementation of the Rosenbrock function and problem.
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_vars_shift.c"
#include "transform_vars_scale.c"
#include "transform_vars_affine.c"
#include "transform_obj_shift.c"
#include "transform_vars_permutation.c"
#include "transform_vars_blockrotation.c"
#include "transform_obj_norm_by_dim.c"

/**
 * @brief Implements the Rosenbrock function without connections to any COCO structures.
 */
static double f_rosenbrock_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double result;
  double s1 = 0.0, s2 = 0.0, tmp;

  assert(number_of_variables > 1);

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  for (i = 0; i < number_of_variables - 1; ++i) {
    tmp = (x[i] * x[i] - x[i + 1]);
    s1 += tmp * tmp;
    tmp = (x[i] - 1.0);
    s2 += tmp * tmp;
  }
  result = 100.0 * s1 + s2;

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_rosenbrock_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_rosenbrock_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Rosenbrock problem.
 */
static coco_problem_t *f_rosenbrock_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Rosenbrock function",
      f_rosenbrock_evaluate, NULL, number_of_variables, -5.0, 5.0, 1.0);
  coco_problem_set_id(problem, "%s_d%02lu", "rosenbrock", number_of_variables);

  /* Compute best solution */
  f_rosenbrock_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Rosenbrock problem.
 */
static coco_problem_t *f_rosenbrock_bbob_problem_allocate(const size_t function,
                                                          const size_t dimension,
                                                          const size_t instance,
                                                          const long rseed,
                                                          const char *problem_id_template,
                                                          const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  size_t i, block_size;
  double *minus_one, factor;

  minus_one = coco_allocate_vector(dimension);
  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  for (i = 0; i < dimension; ++i) {
    minus_one[i] = -1.0;
    xopt[i] *= 0.75;
  }
  fopt = bbob2009_compute_fopt(function, instance);
  if (coco_strfind(problem_name_template, "BBOB large-scale suite") >= 0){
    block_size = coco_rotation_matrix_block_size(dimension);
    factor = coco_double_max(1.0, sqrt((double) block_size) / 8.0);
  } else {
    factor = coco_double_max(1.0, sqrt((double) dimension) / 8.0);
  }

  problem = f_rosenbrock_allocate(dimension);
  problem = transform_vars_shift(problem, minus_one, 0);
  problem = transform_vars_scale(problem, factor);
  problem = transform_vars_shift(problem, xopt, 0);
    
  /*if large scale test-bed, normalize by dim*/
  if (coco_strfind(problem_name_template, "BBOB large-scale suite") >= 0){
        problem = transform_obj_norm_by_dim(problem);
  }
    
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "2-moderate");

  coco_free_memory(minus_one);
  coco_free_memory(xopt);
  return problem;
}

/**
 * @brief Creates the BBOB rotated Rosenbrock problem.
 */
static coco_problem_t *f_rosenbrock_rotated_bbob_problem_allocate(const size_t function,
                                                                  const size_t dimension,
                                                                  const size_t instance,
                                                                  const long rseed,
                                                                  const char *problem_id_template,
                                                                  const char *problem_name_template) {

  double fopt;
  coco_problem_t *problem = NULL;
  size_t row, column;
  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *current_row, **rot1, factor;
  double tmp; /* Wassim: will serve to set the optimal solution "manually"*/

  fopt = bbob2009_compute_fopt(function, instance);
  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed, dimension);

  factor = coco_double_max(1.0, sqrt((double) dimension) / 8.0);
  /* Compute affine transformation */
  for (row = 0; row < dimension; ++row) {
    current_row = M + row * dimension;
    for (column = 0; column < dimension; ++column) {
      current_row[column] = factor * rot1[row][column];
    }
    b[row] = 0.5;
  }
  /*bbob2009_free_matrix(rot1, dimension);*/

  problem = f_rosenbrock_allocate(dimension);
  for (row = 0; row < dimension; row++) {
    problem->best_parameter[row] = 0; /* Wassim: TODO: not a proper way of avoiding to trigger coco_warning("transform_vars_affine(): 'best_parameter' not updated, set to NAN")*/
  }
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_obj_shift(problem, fopt);
  for (column = 0; column < dimension; ++column) { /* Wassim: manually set xopt = rot1^T ones(dimension)/(2*factor) */
    tmp = 0;
    for (row = 0; row < dimension; ++row) {
      tmp += rot1[row][column];
    }
    problem->best_parameter[column] = tmp / (2. * factor);
  }

  bbob2009_free_matrix(rot1, dimension);
  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "2-moderate");

  coco_free_memory(M);
  coco_free_memory(b);
  return problem;
}

/**
 * @brief Creates the BBOB permuted block-rotated Rosenbrock problem.
 */
static coco_problem_t *f_rosenbrock_permblockdiag_bbob_problem_allocate(const size_t function,
                                                                        const size_t dimension,
                                                                        const size_t instance,
                                                                        const long rseed,
                                                                        const char *problem_id_template,
                                                                        const char *problem_name_template) {
  
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  double *minus_one, factor;
  size_t i;

  double **B;
  const double *const *B_copy;
  size_t *P1 = coco_allocate_vector_size_t(dimension);
  size_t *P2 = coco_allocate_vector_size_t(dimension);
  size_t *block_sizes;
  size_t block_size;
  size_t nb_blocks;
  size_t swap_range;
  size_t nb_swaps;
  double *best_parameter = coco_allocate_vector(dimension); /* Manh: will serve to set the optimal solution "manually"*/

  block_sizes = coco_get_block_sizes(&nb_blocks, dimension, "bbob-largescale");
  block_size = coco_rotation_matrix_block_size(dimension);
    
  swap_range = coco_get_swap_range(dimension, "bbob-largescale");
  nb_swaps = coco_get_nb_swaps(dimension, "bbob-largescale");

  fopt = bbob2009_compute_fopt(function, instance);
  factor = coco_double_max(1.0, sqrt((double) block_size) / 8.0);
  minus_one = coco_allocate_vector(dimension);
  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  for (i = 0; i < dimension; ++i) {
      minus_one[i] = -1.0;
      xopt[i] *= 0.75;
  }

  B = coco_allocate_blockmatrix(dimension, block_sizes, nb_blocks);
  B_copy = (const double *const *)B;

  coco_compute_blockrotation(B, rseed, dimension, block_sizes, nb_blocks);
  coco_compute_truncated_uniform_swap_permutation(P1, rseed + 2000000, dimension, nb_swaps, swap_range);
  coco_compute_truncated_uniform_swap_permutation(P2, rseed + 3000000, dimension, nb_swaps, swap_range);

  problem = f_rosenbrock_allocate(dimension);
  problem = transform_vars_shift(problem, minus_one, 0);
  problem = transform_vars_scale(problem, factor);
  problem = transform_vars_permutation(problem, P2, dimension);
  problem = transform_vars_blockrotation(problem, B_copy, dimension, block_sizes, nb_blocks);
  problem = transform_vars_permutation(problem, P1, dimension);

  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_norm_by_dim(problem);
  problem = transform_obj_shift(problem, fopt);
  
  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "2-moderate");

  coco_free_memory(xopt);
  coco_free_memory(best_parameter);
  coco_free_memory(minus_one);
  coco_free_block_matrix(B, dimension);
  coco_free_memory(P1);
  coco_free_memory(P2);
  coco_free_memory(block_sizes);
  return problem;
}


