/**
 * @file f_step_ellipsoid.c
 * @brief Implementation of the step ellipsoid function and problem.
 *
 * The BBOB step ellipsoid function intertwines the variable and objective transformations in such a way
 * that it is hard to devise a composition of generic transformations to implement it. In the end one would
 * have to implement several custom transformations which would be used solely by this problem. Therefore
 * we opt to implement it as a monolithic function instead.
 *
 * TODO: It would be nice to have a generic step ellipsoid function to complement this one.
 */
#include <assert.h>
#include <stdio.h>

#include "coco.h"
#include "coco_problem.c"
#include "coco_utilities.c"
#include "suite_bbob_legacy_code.c"

#include "transform_obj_penalize.c"
#include "transform_obj_shift.c"
#include "transform_vars_shift.c"
#include "transform_vars_permutation.c"
#include "transform_vars_blockrotation.c"
#include "transform_vars_round_step.c"
#include "transform_obj_norm_by_dim.c"


/**
 * @brief Data type for the step ellipsoid problem.
 */
typedef struct {
  double *x, *xx;
  double *xopt, fopt, penalty_scale;
  double **rot1, **rot2;
} f_step_ellipsoid_data_t;

/**
 * @brief Implements the step ellipsoid function without connections to any COCO structures.
 */
static double f_step_ellipsoid_raw(const double *x, const size_t number_of_variables, const f_step_ellipsoid_data_t *data) {
  
  static const double condition = 100;
  static const double alpha = 10.0;
  size_t i, j;
  double penalty = 0.0, x1;
  double result;
  
  assert(number_of_variables > 1);

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  for (i = 0; i < number_of_variables; ++i) {
    double tmp;
    tmp = fabs(x[i]) - 5.0;
    if (tmp > 0.0)
      penalty += tmp * tmp;
  }
  penalty = penalty * (data -> penalty_scale);
  for (i = 0; i < number_of_variables; ++i) {
    double c1;
    data->x[i] = 0.0;
    c1 = sqrt(pow(condition / 10., (double) i / (double) (number_of_variables - 1)));
    for (j = 0; j < number_of_variables; ++j) {
      data->x[i] += c1 * data->rot2[i][j] * (x[j] - data->xopt[j]);
    }
  }
  x1 = data->x[0];
  
  for (i = 0; i < number_of_variables; ++i) {
    if (fabs(data->x[i]) > 0.5) /* TODO: Documentation: no fabs() in documentation */
      data->x[i] = coco_double_round(data->x[i]);
    else
      data->x[i] = coco_double_round(alpha * data->x[i]) / alpha;
  }
  
  for (i = 0; i < number_of_variables; ++i) {
    data->xx[i] = 0.0;
    for (j = 0; j < number_of_variables; ++j) {
      data->xx[i] += data->rot1[i][j] * data->x[j];
    }
  }
  
  /* Computation core */
  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    double exponent;
    exponent = (double) (long) i / ((double) (long) number_of_variables - 1.0);
    result += pow(condition, exponent) * data->xx[i] * data->xx[i];
  }
  result = 0.1 * coco_double_max(fabs(x1) * 1.0e-4, result) + penalty + data->fopt;
  
  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_step_ellipsoid_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_step_ellipsoid_raw(x, problem->number_of_variables, (f_step_ellipsoid_data_t *) problem->data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the step ellipsoid data object.
 */
static void f_step_ellipsoid_free(coco_problem_t *problem) {
  f_step_ellipsoid_data_t *data;
  data = (f_step_ellipsoid_data_t *) problem->data;
  coco_free_memory(data->x);
  coco_free_memory(data->xx);
  coco_free_memory(data->xopt);
  bbob2009_free_matrix(data->rot1, problem->number_of_variables);
  bbob2009_free_matrix(data->rot2, problem->number_of_variables);
  /* Let the generic free problem code deal with all of the coco_problem_t fields */
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Creates the BBOB step ellipsoid problem.
 *
 * @note There is no separate basic allocate function.
 */
static coco_problem_t *f_step_ellipsoid_bbob_problem_allocate(const size_t function,
                                                              const size_t dimension,
                                                              const size_t instance,
                                                              const long rseed,
                                                              const void *args,
                                                              const char *problem_id_template,
                                                              const char *problem_name_template) {
  
  f_step_ellipsoid_data_t *data;
  size_t i;
  coco_problem_t *problem = coco_problem_allocate_from_scalars("step ellipsoid function",
                                                               f_step_ellipsoid_evaluate, f_step_ellipsoid_free, dimension, -5.0, 5.0, 0);
  
  data = (f_step_ellipsoid_data_t *) coco_allocate_memory(sizeof(*data));
  /* Allocate temporary storage and space for the rotation matrices */
  data->x = coco_allocate_vector(dimension);
  data->xx = coco_allocate_vector(dimension);
  data->xopt = coco_allocate_vector(dimension);
  data->rot1 = bbob2009_allocate_matrix(dimension, dimension);
  data->rot2 = bbob2009_allocate_matrix(dimension, dimension);

  f_step_ellipsoid_args_t *f_step_ellipsoid_args;
  f_step_ellipsoid_args = ((f_step_ellipsoid_args_t *) args);

  double penalty_scale = f_step_ellipsoid_args->penalty_scale;
  data->fopt = bbob2009_compute_fopt(function, instance);
  data->penalty_scale = penalty_scale;
  bbob2009_compute_xopt(data->xopt, rseed, dimension);
  bbob2009_compute_rotation(data->rot1, rseed + 1000000, dimension);
  bbob2009_compute_rotation(data->rot2, rseed, dimension);
  
  problem->data = data;
  
  /* Compute best solution
   *
   * OME: Dirty hack for now because I did not want to invert the
   * transformations to find the best_parameter :/
   */
  for (i = 0; i < problem->number_of_variables; i++) {
    problem->best_parameter[i] = data->xopt[i];
  }
  problem->best_value[0] = data->fopt;
  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "2-moderate");
  
  return problem;
}

/**
 * @brief Implements the step ellipsoid function without connections to any COCO structures.
 */
static double f_step_ellipsoid_core(const double *x, const size_t number_of_variables, f_step_ellipsoid_versatile_data_t *f_step_ellipsoid_versatile_data) {
  
  static const double condition = 100;
  size_t i;
  double result;
  result = 0.0;
  
  for (i = 0; i < number_of_variables; ++i) {
    double exponent;
    exponent = (double) (long) i / ((double) (long) number_of_variables - 1.0);
    result += pow(condition, exponent) * x[i] * x[i];
  }
  result = 0.1 * coco_double_max(f_step_ellipsoid_versatile_data->zhat_1 * 1.0e-4, result);
  return result;
}

/**
 * @brief Uses the raw function to evaluate the ls COCO problem.
 */
static void f_step_ellipsoid_permblock_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_step_ellipsoid_core(x, problem->number_of_variables, (f_step_ellipsoid_versatile_data_t *) problem->versatile_data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief allows to free the versatile_data part of the problem.
 */
static void f_step_ellipsoid_versatile_data_free(coco_problem_t *problem) {
  coco_free_memory((f_step_ellipsoid_versatile_data_t *) problem->versatile_data);
  problem->versatile_data = NULL;
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Allocates the basic step ellipsoid problem.
 * an additional coordinate is added that will contain the value of \hat{z}_1 but that is ignored by functions other that f_step_ellipsoid_core and transform_vars_round_step. The latter sets it.
 */
static coco_problem_t *f_step_ellipsoid_allocate(const size_t number_of_variables) {
  
  coco_problem_t *problem = coco_problem_allocate_from_scalars("step ellipsoid function",
                                                               f_step_ellipsoid_permblock_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  problem->versatile_data = (f_step_ellipsoid_versatile_data_t *) coco_allocate_memory(sizeof(f_step_ellipsoid_versatile_data_t));
  ((f_step_ellipsoid_versatile_data_t *) problem->versatile_data)->zhat_1 = 0;/*needed for xopt evaluation*/
  /* add the free function of the allocated versatile_data*/
  problem->problem_free_function = f_step_ellipsoid_versatile_data_free;
  
  coco_problem_set_id(problem, "%s_d%02lu", "step_ellipsoid", number_of_variables);
  /* Compute best solution, here done outside after the zhat is set to the best_value */
  f_step_ellipsoid_permblock_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB permuted block-rotated step ellipsoid problem.
 *
 * Wassim: TODO: consider implementing it sub-problem style
 * Wassim: TODO: make the zhat1 value default to x1 when no transformation is applied and the data type defined here
 */
static coco_problem_t *f_step_ellipsoid_permblockdiag_bbob_problem_allocate(const size_t function,
                                                                            const size_t dimension,
                                                                            const size_t instance,
                                                                            const long rseed,
                                                                            const char *problem_id_template,
                                                                            const char *problem_name_template) {
  double alpha = 10.; /*parameter of rounding*/
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  double **B1, **B2;
  const double *const *B1_copy;
  const double *const *B2_copy;
  size_t *P11, *P12, *P21, *P22;
  size_t *block_sizes1, *block_sizes2, nb_blocks1, nb_blocks2, swap_range1, swap_range2, nb_swaps1, nb_swaps2;
  double penalty_factor = 1.;
  
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
  B1_copy = (const double *const *)B1;
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

  problem = f_step_ellipsoid_allocate(dimension);

  problem = transform_vars_permutation(problem, P22, dimension);
  problem = transform_vars_blockrotation(problem, B1_copy, dimension, block_sizes1, nb_blocks1);
  problem = transform_vars_permutation(problem, P21, dimension);
  problem = transform_vars_round_step(problem, alpha);
  
  problem = transform_vars_conditioning(problem, 10.0);
  problem = transform_vars_permutation(problem, P12, dimension);
  problem = transform_vars_blockrotation(problem, B2_copy, dimension, block_sizes2, nb_blocks2);
  problem = transform_vars_permutation(problem, P11, dimension);
  problem = transform_vars_shift(problem, xopt, 0);

  problem = transform_obj_norm_by_dim(problem);
  problem = transform_obj_penalize(problem, penalty_factor);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "2-moderate");

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
