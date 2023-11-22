/**
 * @file f_schaffers.c
 * @brief Implementation of the Schaffer's F7 function and problem, transformations not implemented for the
 * moment.
 */

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_shift.c"
#include "transform_vars_asymmetric.c"
#include "transform_vars_affine.c"
#include "transform_vars_shift.c"
#include "transform_obj_penalize.c"
#include "transform_vars_permutation.c"
#include "transform_vars_blockrotation.c"
#include "transform_obj_norm_by_dim.c"

/**
 * @brief Data type for the schaffers problem
 */
typedef struct{
  double conditioning;
  double penalty_scale;
} f_schaffers_data_t;

/**
 * @brief Implements the Schaffer's F7 function without connections to any COCO structures.
 */
static double f_schaffers_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double result;

  assert(number_of_variables > 1);

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  /* Computation core */
  result = 0.0;
  for (i = 0; i < number_of_variables - 1; ++i) {
    const double tmp = x[i] * x[i] + x[i + 1] * x[i + 1];
    if (coco_is_inf(tmp) && coco_is_nan(sin(50.0 * pow(tmp, 0.1))))  /* sin(inf) -> nan */
      /* the second condition is necessary to pass the integration tests under Windows and Linux */
      return tmp;
    result += pow(tmp, 0.25) * (1.0 + pow(sin(50.0 * pow(tmp, 0.1)), 2.0));
  }
  result = pow(result / ((double) (long) number_of_variables - 1.0), 2.0);

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_schaffers_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_schaffers_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Schaffer's F7 problem.
 */
static coco_problem_t *f_schaffers_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Schaffer's function",
      f_schaffers_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "schaffers", number_of_variables);

  /* Compute best solution */
  f_schaffers_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Schaffer's F7 problem.
 */
static coco_problem_t *f_schaffers_bbob_problem_allocate(const size_t function,
                                                         const size_t dimension,
                                                         const size_t instance,
                                                         const long rseed,
                                                         const void *args,
                                                         const char *problem_id_template,
                                                         const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  size_t i, j;
  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *current_row, **rot1, **rot2;

  f_schaffers_args_t *f_schaffers_args;
  f_schaffers_args = ((f_schaffers_args_t *) args);
  f_schaffers_data_t *data;
  data = (f_schaffers_data_t *) coco_allocate_memory(sizeof(*data));

  data->conditioning = f_schaffers_args->conditioning;
  data->penalty_scale = f_schaffers_args->penalty_scale;

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
      double exponent = 1.0 * (int) i / ((double) (long) dimension - 1.0);
      current_row[j] = rot2[i][j] * pow(sqrt(data -> conditioning), exponent);
    }
  }

  problem = f_schaffers_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_asymmetric(problem, 0.5);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_penalize(problem, data->penalty_scale);

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
 * @brief Creates the BBOB permuted block-rotated Schaffer's F7 problem.
 */
static coco_problem_t *f_schaffers_permblockdiag_bbob_problem_allocate(const size_t function,
                                                                       const size_t dimension,
                                                                       const size_t instance,
                                                                       const long rseed,
                                                                       const double conditioning,
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

    const double penalty_factor = 10.0;

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
    coco_compute_blockrotation(B2, rseed, dimension, block_sizes2, nb_blocks2);
    
    coco_compute_truncated_uniform_swap_permutation(P11, rseed + 3000000, dimension, nb_swaps, swap_range);
    coco_compute_truncated_uniform_swap_permutation(P21, rseed + 4000000, dimension, nb_swaps, swap_range);
    coco_compute_truncated_uniform_swap_permutation(P12, rseed + 5000000, dimension, nb_swaps, swap_range);
    coco_compute_truncated_uniform_swap_permutation(P22, rseed + 6000000, dimension, nb_swaps, swap_range);
    
    problem = f_schaffers_allocate(dimension);
    problem = transform_vars_conditioning(problem, conditioning);
    problem = transform_vars_permutation(problem, P21, dimension);
    problem = transform_vars_blockrotation(problem, B2_copy, dimension, block_sizes2, nb_blocks2);
    problem = transform_vars_permutation(problem, P11, dimension);
    
    problem = transform_vars_asymmetric(problem, 0.5);
    problem = transform_vars_permutation(problem, P22, dimension);
    problem = transform_vars_blockrotation(problem, B1_copy, dimension, block_sizes1, nb_blocks1);
    problem = transform_vars_permutation(problem, P12, dimension);
    
    problem = transform_vars_shift(problem, xopt, 0);
    /*problem = transform_obj_norm_by_dim(problem);*/ /* Wassim: there is already a normalization by dimension*/
    problem = transform_obj_penalize(problem, penalty_factor);
    problem = transform_obj_shift(problem, fopt);
    
    coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
    coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
    coco_problem_set_type(problem, "4-multi-modal");
    
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

