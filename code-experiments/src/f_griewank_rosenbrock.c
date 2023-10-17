/**
 * @file f_griewank_rosenbrock.c
 * @brief Implementation of the Griewank-Rosenbrock function and problem.
 */

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_vars_affine.c"
#include "transform_vars_shift.c"
#include "transform_obj_shift.c"
#include "transform_vars_scale.c"
#include "transform_vars_permutation.c"
#include "transform_vars_blockrotation.c"
#include "transform_obj_norm_by_dim.c"

/**
 * @brief Data type for the griewank rosenbrock problem
 */
typedef struct{
  double facftrue;
} f_griewank_rosenbrock_data_t;

/**
 * @brief Implements the Griewank-Rosenbrock function without connections to any COCO structures.
 */
static double f_griewank_rosenbrock_raw(const double *x, const size_t number_of_variables, f_griewank_rosenbrock_data_t * data) {

  size_t i = 0;
  double tmp = 0;
  double result;

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  /* Computation core */
  result = 0.0;
  for (i = 0; i < number_of_variables - 1; ++i) {
    const double c1 = x[i] * x[i] - x[i + 1];
    const double c2 = 1.0 - x[i];
    tmp = 100.0 * c1 * c1 + c2 * c2;
    result += tmp / 4000. - cos(tmp);
  }
  result = data->facftrue + data->facftrue * result / (double) (number_of_variables - 1);

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_griewank_rosenbrock_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_griewank_rosenbrock_raw(x, problem->number_of_variables, (f_griewank_rosenbrock_data_t *) problem -> data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Griewank-Rosenbrock problem.
 */
static coco_problem_t *f_griewank_rosenbrock_allocate(const size_t number_of_variables, double facftrue) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Griewank Rosenbrock function",
      f_griewank_rosenbrock_evaluate, NULL, number_of_variables, -5.0, 5.0, 1);
  coco_problem_set_id(problem, "%s_d%02lu", "griewank_rosenbrock", number_of_variables);


  f_griewank_rosenbrock_data_t *data;
  data = (f_griewank_rosenbrock_data_t *) coco_allocate_memory(sizeof(*data));
  data->facftrue = facftrue;
  problem->data = data;
  /* Compute best solution */
  f_griewank_rosenbrock_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Griewank-Rosenbrock problem.
 */
static coco_problem_t *f_griewank_rosenbrock_bbob_problem_allocate(const size_t function,
                                                                   const size_t dimension,
                                                                   const size_t instance,
                                                                   const long rseed,
                                                                  const f_args_t *args,
                                                                   const char *problem_id_template,
                                                                   const char *problem_name_template) {
  double fopt;
  coco_problem_t *problem = NULL;
  size_t i, j;

  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *shift = coco_allocate_vector(dimension);
  double scales, **rot1;
  double tmp; /* Wassim: will serve to set the optimal solution "manually"*/

  fopt = bbob2009_compute_fopt(function, instance);
  for (i = 0; i < dimension; ++i) {
    shift[i] = -0.5;
  }

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed, dimension);
  scales = coco_double_max(1., sqrt((double) dimension) / 8.);
  for (i = 0; i < dimension; ++i) {
    for (j = 0; j < dimension; ++j) {
      rot1[i][j] *= scales;
    }
  }

  problem = f_griewank_rosenbrock_allocate(dimension, args->f_griewank_rosenbrock_args->facftrue);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_shift(problem, shift, 0);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);

  for (i = 0; i < dimension; i++) {
    problem->best_parameter[i] = 0; /* Wassim: TODO: not a proper way of avoiding to trigger coco_warning("transform_vars_affine(): 'best_parameter' not updated, set to NAN")*/
  }
  problem = transform_vars_affine(problem, M, b, dimension);
  for (j = 0; j < dimension; ++j) { /* Wassim: manually set xopt = rot1^T ones(dimension)/(2*factor) */
    tmp = 0;
    for (i = 0; i < dimension; ++i) {
      tmp += rot1[i][j];
    }
    problem->best_parameter[j] = tmp / (2. * scales);
  }
  bbob2009_free_matrix(rot1, dimension);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "4-multi-modal");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(shift);
  return problem;
}

/**
 * @brief Creates the BBOB permuted block-rotated Griewank-Rosenbrock problem.
 */
static coco_problem_t *f_griewank_rosenbrock_permblockdiag_bbob_bbob_problem_allocate(const size_t function,
                                                                                      const size_t dimension,
                                                                                      const size_t instance,
                                                                                      const long rseed,
                                                                                      const f_args_t *args,
                                                                                      const char *problem_id_template,
                                                                                      const char *problem_name_template) {
  double fopt;
  coco_problem_t *problem = NULL;
  double *shift, scales;
  size_t i, j, k, next_bs_change;
  
  double **B;
  const double *const *B_copy;
  size_t *P1 = coco_allocate_vector_size_t(dimension);
  size_t *P2 = coco_allocate_vector_size_t(dimension);
  size_t *block_sizes;
  size_t block_size;
  size_t nb_blocks;
  size_t swap_range;
  size_t nb_swaps;
  double tmp; /* Manh: will serve to set the optimal solution "manually"*/
  double *best_parameter = coco_allocate_vector(dimension); /* Manh: will serve to set the optimal solution "manually"*/

  
  block_sizes = coco_get_block_sizes(&nb_blocks, dimension, "bbob-largescale");
  block_size = coco_rotation_matrix_block_size(dimension);
  swap_range = coco_get_swap_range(dimension, "bbob-largescale");
  nb_swaps = coco_get_nb_swaps(dimension, "bbob-largescale");
  
  fopt = bbob2009_compute_fopt(function, instance);
  scales = coco_double_max(1.0, sqrt((double) block_size) / 8.0);
  shift = coco_allocate_vector(dimension);
  for (i = 0; i < dimension; ++i) {
      shift[i] = -0.5;
  }
  
  B = coco_allocate_blockmatrix(dimension, block_sizes, nb_blocks);
  B_copy = (const double *const *)B;
  
  coco_compute_blockrotation(B, rseed, dimension, block_sizes, nb_blocks);
  coco_compute_truncated_uniform_swap_permutation(P1, rseed + 2000000, dimension, nb_swaps, swap_range);
  coco_compute_truncated_uniform_swap_permutation(P2, rseed + 3000000, dimension, nb_swaps, swap_range);
  
  problem = f_griewank_rosenbrock_allocate(dimension, args->f_griewank_rosenbrock_args->facftrue);
  problem = transform_vars_shift(problem, shift, 0);
  problem = transform_vars_scale(problem, scales);
  problem = transform_vars_permutation(problem, P2, dimension);
  problem = transform_vars_blockrotation(problem, B_copy, dimension, block_sizes, nb_blocks);
  problem = transform_vars_permutation(problem, P1, dimension);
  
  /*problem = transform_obj_norm_by_dim(problem);*/ /* Wassim: there is already a normalization by dimension*/
  problem = transform_obj_shift(problem, fopt);

  /* Manh: manually set xopt = rot1^T ones(dimension)/(2*scales) */
  next_bs_change = 0;
  for (k = 0; k < nb_blocks; ++k){
    for (j = 0; j < block_sizes[k]; ++j) { /* Manh: firstly, set xopt_1 = (B^T)*(P_2^T)*ones(dimension)/(2*scales) */
      tmp = 0;
      for (i = 0; i < block_sizes[k]; ++i) {
        tmp += B[next_bs_change + i][j];
      }
      best_parameter[next_bs_change + j] = tmp / (2. * scales);
    }
    next_bs_change += block_sizes[k];
  }

  for (j = 0; j < dimension; ++j) { /* Manh: secondly, set xopt = (P_1^T)* xopt_1 */
    problem->best_parameter[P1[j]] = best_parameter[j];
  }

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "4-multi-modal");
  
  coco_free_memory(best_parameter);
  coco_free_memory(shift);
  coco_free_block_matrix(B, dimension);
  coco_free_memory(P1);
  coco_free_memory(P2);
  coco_free_memory(block_sizes);
  return problem;
}

