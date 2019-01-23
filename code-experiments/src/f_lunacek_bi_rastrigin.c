/**
 * @file f_lunacek_bi_rastrigin.c
 * @brief Implementation of the Lunacek bi-Rastrigin function and problem.
 */

#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_shift.c"
#include "f_sphere.c"
#include "transform_vars_x_hat_generic.c"
#include "transform_obj_norm_by_dim.c"
#include "transform_obj_scale.c"

/**
 * @brief Data type for the Lunacek bi-Rastrigin problem.
 */
typedef struct {
  double *x_hat, *z;
  double *xopt, fopt;
  double **rot1, **rot2;
  long rseed;
  coco_problem_free_function_t old_free_problem;
} f_lunacek_bi_rastrigin_data_t;

/**
 * @brief Implements the Lunacek bi-Rastrigin function without connections to any COCO structures.
 */
static double f_lunacek_bi_rastrigin_raw(const double *x,
                                         const size_t number_of_variables,
                                         f_lunacek_bi_rastrigin_data_t *data) {
  double result;
  static const double condition = 100.;
  size_t i, j;
  double penalty = 0.0;
  static const double mu0 = 2.5;
  static const double d = 1.;
  const double s = 1. - 0.5 / (sqrt((double) (number_of_variables + 20)) - 4.1);
  const double mu1 = -sqrt((mu0 * mu0 - d) / s);
  double *tmpvect, sum1 = 0., sum2 = 0., sum3 = 0.;

  assert(number_of_variables > 1);

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  for (i = 0; i < number_of_variables; ++i) {
    double tmp;
    tmp = fabs(x[i]) - 5.0;
    if (tmp > 0.0)
      penalty += tmp * tmp;
  }

  /* x_hat */
  for (i = 0; i < number_of_variables; ++i) {
    data->x_hat[i] = 2. * x[i];
    if (data->xopt[i] < 0.) {
      data->x_hat[i] *= -1.;
    }
  }

  tmpvect = coco_allocate_vector(number_of_variables);
  /* affine transformation */
  for (i = 0; i < number_of_variables; ++i) {
    double c1;
    tmpvect[i] = 0.0;
    c1 = pow(sqrt(condition), ((double) i) / (double) (number_of_variables - 1));
    for (j = 0; j < number_of_variables; ++j) {
      tmpvect[i] += c1 * data->rot2[i][j] * (data->x_hat[j] - mu0);
    }
  }
  for (i = 0; i < number_of_variables; ++i) {
    data->z[i] = 0;
    for (j = 0; j < number_of_variables; ++j) {
      data->z[i] += data->rot1[i][j] * tmpvect[j];
    }
  }
  /* Computation core */
  for (i = 0; i < number_of_variables; ++i) {
    sum1 += (data->x_hat[i] - mu0) * (data->x_hat[i] - mu0);
    sum2 += (data->x_hat[i] - mu1) * (data->x_hat[i] - mu1);
    sum3 += cos(2 * coco_pi * data->z[i]);
  }
  result = coco_double_min(sum1, d * (double) number_of_variables + s * sum2)
      + 10. * ((double) number_of_variables - sum3) + 1e4 * penalty;
  coco_free_memory(tmpvect);

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_lunacek_bi_rastrigin_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_lunacek_bi_rastrigin_raw(x, problem->number_of_variables, (f_lunacek_bi_rastrigin_data_t *) problem->data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the Lunacek bi-Rastrigin data object.
 */
static void f_lunacek_bi_rastrigin_free(coco_problem_t *problem) {
  f_lunacek_bi_rastrigin_data_t *data;
  data = (f_lunacek_bi_rastrigin_data_t *) problem->data;
  coco_free_memory(data->x_hat);
  coco_free_memory(data->z);
  coco_free_memory(data->xopt);
  bbob2009_free_matrix(data->rot1, problem->number_of_variables);
  bbob2009_free_matrix(data->rot2, problem->number_of_variables);

  /* Let the generic free problem code deal with all of the
   * coco_problem_t fields.
   */
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Creates the BBOB Lunacek bi-Rastrigin problem.
 *
 * @note There is no separate basic allocate function.
 */
static coco_problem_t *f_lunacek_bi_rastrigin_bbob_problem_allocate(const size_t function,
                                                                    const size_t dimension,
                                                                    const size_t instance,
                                                                    const long rseed,
                                                                    const char *problem_id_template,
                                                                    const char *problem_name_template) {

  f_lunacek_bi_rastrigin_data_t *data;
  coco_problem_t *problem = coco_problem_allocate_from_scalars("Lunacek's bi-Rastrigin function",
      f_lunacek_bi_rastrigin_evaluate, f_lunacek_bi_rastrigin_free, dimension, -5.0, 5.0, 0.0);

  const double mu0 = 2.5;

  double fopt, *tmpvect;
  size_t i;

  data = (f_lunacek_bi_rastrigin_data_t *) coco_allocate_memory(sizeof(*data));
  /* Allocate temporary storage and space for the rotation matrices */
  data->x_hat = coco_allocate_vector(dimension);
  data->z = coco_allocate_vector(dimension);
  data->xopt = coco_allocate_vector(dimension);
  data->rot1 = bbob2009_allocate_matrix(dimension, dimension);
  data->rot2 = bbob2009_allocate_matrix(dimension, dimension);
  data->rseed = rseed;

  data->fopt = bbob2009_compute_fopt(24, instance);
  bbob2009_compute_xopt(data->xopt, rseed, dimension);
  bbob2009_compute_rotation(data->rot1, rseed + 1000000, dimension);
  bbob2009_compute_rotation(data->rot2, rseed, dimension);

  problem->data = data;

  /* Compute best solution */
  tmpvect = coco_allocate_vector(dimension);
  bbob2009_gauss(tmpvect, dimension, rseed);
  for (i = 0; i < dimension; ++i) {
    data->xopt[i] = 0.5 * mu0;
    if (tmpvect[i] < 0.0) {
      data->xopt[i] *= -1.0;
    }
    problem->best_parameter[i] = data->xopt[i];
  }
  coco_free_memory(tmpvect);
  f_lunacek_bi_rastrigin_evaluate(problem, problem->best_parameter, problem->best_value);

  fopt = bbob2009_compute_fopt(function, instance);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "5-weakly-structured");

  return problem;
}



/* Functions used in/related to the large scale suite are below. Eventually either merge with above after the standard version is updated with the new approach or put what's below in a separate file*/



/**
 * @brief allows to free the versatile_data part of the problem.
 */
static void f_lunacek_bi_rastrigin_versatile_data_free(coco_problem_t *problem) {

  f_lunacek_bi_rastrigin_versatile_data_t *versatile_data = (f_lunacek_bi_rastrigin_versatile_data_t *) problem->versatile_data;
  /*free the two problems*/
  if (versatile_data->sub_problem_mu0 != NULL) {
    coco_problem_free(versatile_data->sub_problem_mu0);
  }
  if (versatile_data->sub_problem_mu1 != NULL) {
    coco_problem_free(versatile_data->sub_problem_mu1);
  }

  coco_free_memory(versatile_data->x_hat);  /*Manh: free the x_hat*/
  coco_free_memory(versatile_data);
  problem->versatile_data = NULL;
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}


/**
 * @brief Uses the core function to evaluate the sub problem.
 */
static void f_lunacek_bi_rastrigin_sub_evaluate_core(coco_problem_t *problem, const double *x, double *y) {
  
  assert(problem->number_of_objectives == 1);
  y[0] = f_sphere_raw(x, problem->number_of_variables);
}


/**
 * @brief Allocates the basic lunacek_bi_rastrigin sub problem.
 */
static coco_problem_t *f_lunacek_bi_rastrigin_sub_problem_allocate(const size_t number_of_variables) {
  
  coco_problem_t *problem_i = coco_problem_allocate_from_scalars("lunacek_bi_rastrigin_sub function",
                                                                 f_lunacek_bi_rastrigin_sub_evaluate_core, NULL, number_of_variables, -5.0, 5.0, 0.0);
  problem_i->versatile_data = NULL;
  coco_problem_set_id(problem_i, "%s_d%04lu", "lunacek_bi_rastrigin_sub", number_of_variables);
  f_lunacek_bi_rastrigin_sub_evaluate_core(problem_i, problem_i->best_parameter, problem_i->best_value);
  return problem_i;
}


/**
 * @brief Implements the lunacek_bi_rastrigin function without connections to any COCO structures.
 * Wassim: core to not conflict with raw for now
 */
static double f_lunacek_bi_rastrigin_core(const double *x, const size_t number_of_variables,f_lunacek_bi_rastrigin_versatile_data_t *f_lunacek_bi_rastrigin_versatile_data) {

  coco_problem_t *problem_sub_mu0, *problem_sub_mu1;
  size_t i;
  double result = 0.0;
  double y0, y1;
  double *x_hat;

  x_hat = f_lunacek_bi_rastrigin_versatile_data->x_hat;

  problem_sub_mu0 = f_lunacek_bi_rastrigin_versatile_data->sub_problem_mu0;
  problem_sub_mu0->evaluate_function(problem_sub_mu0, x_hat, &y0);

  problem_sub_mu1 = f_lunacek_bi_rastrigin_versatile_data->sub_problem_mu1;
  problem_sub_mu1->evaluate_function(problem_sub_mu1, x_hat, &y1);

  result += (double) number_of_variables;

  for (i = 0; i < number_of_variables; i++) {
    result -= cos(2 * coco_pi * x[i]);
  }
  result *= 10.0;
  result += coco_double_min(y0, y1);

  return result;
}

/**
 * @brief Uses the core function to evaluate the COCO problem.
 */
static void f_lunacek_bi_rastrigin_evaluate_core(coco_problem_t *problem, const double *x, double *y) {
  
  assert(problem->number_of_objectives == 1);
  y[0] = f_lunacek_bi_rastrigin_core(x, problem->number_of_variables, ((f_lunacek_bi_rastrigin_versatile_data_t *) problem->versatile_data));
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic lunacek_bi_rastrigin problem.
 */
static coco_problem_t *f_lunacek_bi_rastrigin_problem_allocate(const size_t number_of_variables) {
  
  coco_problem_t *problem = coco_problem_allocate_from_scalars("lunacek_bi_rastrigin function",
                                                               f_lunacek_bi_rastrigin_evaluate_core, f_lunacek_bi_rastrigin_versatile_data_free, number_of_variables, -5.0, 5.0, 0.0);
    
  problem->versatile_data = (f_lunacek_bi_rastrigin_versatile_data_t *) coco_allocate_memory(sizeof(f_lunacek_bi_rastrigin_versatile_data_t));
  ((f_lunacek_bi_rastrigin_versatile_data_t *) problem->versatile_data)->x_hat = coco_allocate_vector(number_of_variables); /* Manh: Allocate x_hat in versatile_data */
  coco_problem_set_id(problem, "%s_d%04lu", "lunacek_bi_rastrigin", number_of_variables);
  /* Compute the best solution later once the sub-problems are well defined */
  *(problem->best_value) = 0;  /* Manh: set default value to avoid assert() in transformation later*/
  return problem;
}

/**
 * @brief Creates the BBOB large scale suite Lunacek bi-Rastrigin problem.
 */
static coco_problem_t *f_lunacek_bi_rastrigin_permblockdiag_bbob_problem_allocate(const size_t function,
                                                                       const size_t dimension,
                                                                       const size_t instance,
                                                                       const long rseed,
                                                                       const char *problem_id_template,
                                                                       const char *problem_name_template) {
  size_t i;
  double fopt;
  double penalty_factor = 1e4;
  coco_problem_t *problem = NULL, **sub_problem_tmp;

  double **B1, **B2;
  const double *const *B1_copy;
  const double *const *B2_copy;
  size_t *P11, *P12, *P21, *P22;
  size_t *block_sizes1, *block_sizes2;
  size_t nb_blocks1, nb_blocks2;
  size_t swap_range1, swap_range2;
  size_t nb_swaps1, nb_swaps2;

  double condition = 100.0;
  double mu0, mu1, d = 1.0, s;
  double *mu0_vector, *mu1_vector, *sign_vector; /* Wassim sign vector designate the 1^+_- vector*/
  coco_random_state_t *rng = coco_random_new((uint32_t) rseed);

  fopt = bbob2009_compute_fopt(function, instance);
  s = 1. - 0.5 / (sqrt((double) (dimension + 20)) - 4.1);
  mu0 = 2.5;
  mu1 = -sqrt((mu0 * mu0 - d) / s);

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
  coco_compute_blockrotation(B2, rseed + 2000000, dimension, block_sizes2, nb_blocks2);

  P11 = coco_allocate_vector_size_t(dimension);
  P12 = coco_allocate_vector_size_t(dimension);
  P21 = coco_allocate_vector_size_t(dimension);
  P22 = coco_allocate_vector_size_t(dimension);
  coco_compute_truncated_uniform_swap_permutation(P11, rseed + 3000000, dimension, nb_swaps1, swap_range1);
  coco_compute_truncated_uniform_swap_permutation(P12, rseed + 4000000, dimension, nb_swaps1, swap_range1);
  coco_compute_truncated_uniform_swap_permutation(P21, rseed + 5000000, dimension, nb_swaps2, swap_range2);
  coco_compute_truncated_uniform_swap_permutation(P22, rseed + 6000000, dimension, nb_swaps2, swap_range2);

  problem = f_lunacek_bi_rastrigin_problem_allocate(dimension);

  mu0_vector = coco_allocate_vector(dimension);
  mu1_vector = coco_allocate_vector(dimension);
  for (i = 0; i < dimension; i++) {
    mu0_vector[i] = mu0;
    mu1_vector[i] = mu1;
  }
  /* allocate sub-problems */
  ((f_lunacek_bi_rastrigin_versatile_data_t *) problem->versatile_data)->sub_problem_mu0 = f_lunacek_bi_rastrigin_sub_problem_allocate(dimension);
  ((f_lunacek_bi_rastrigin_versatile_data_t *) problem->versatile_data)->sub_problem_mu1 = f_lunacek_bi_rastrigin_sub_problem_allocate(dimension);

  /* set sign_vector */
  sign_vector = coco_allocate_vector(dimension);
  for ( i = 0; i < dimension; i++) { /* set sign(x_opt)*/
    if ( coco_random_normal(rng) < 0.0) {/* Wassim: noraml is used here but unif for Schweffel!!! */
      sign_vector[i] = -1.0;
    } else {
      sign_vector[i] = 1.0;
    }
  }

  /* apply transformations to sub-problems */
  sub_problem_tmp = &((f_lunacek_bi_rastrigin_versatile_data_t *) problem->versatile_data)->sub_problem_mu0;
  *sub_problem_tmp = transform_vars_shift(*sub_problem_tmp, mu0_vector, 0);

  sub_problem_tmp = &((f_lunacek_bi_rastrigin_versatile_data_t *) problem->versatile_data)->sub_problem_mu1;
  *sub_problem_tmp = transform_vars_shift(*sub_problem_tmp, mu1_vector, 0);
  *sub_problem_tmp = transform_obj_scale(*sub_problem_tmp, s);  /* Manh: need to use s */
  *sub_problem_tmp = transform_obj_shift(*sub_problem_tmp, d * (double) dimension);
  
  /* transformations on main problem */
  problem = transform_vars_permutation(problem, P22, dimension);
  problem = transform_vars_blockrotation(problem, B2_copy, dimension, block_sizes2, nb_blocks2);
  problem = transform_vars_permutation(problem, P21, dimension);
  problem = transform_vars_conditioning(problem, condition);
  
  problem = transform_vars_permutation(problem, P12, dimension);
  problem = transform_vars_blockrotation(problem, B1_copy, dimension, block_sizes1, nb_blocks1);
  problem = transform_vars_permutation(problem, P11, dimension);
  problem = transform_vars_shift(problem, mu0_vector, 0);
  problem = transform_vars_x_hat_generic(problem, sign_vector);

  problem = transform_obj_norm_by_dim(problem);
  problem = transform_obj_penalize(problem, penalty_factor);
  problem = transform_obj_shift(problem, fopt);

  /* set best_parameter and best value*/
  for ( i = 0; i < dimension; i++) { /* Wassim: to silence warning about best_parameter*/
    problem->best_parameter[i] = 0.5 * mu0 * sign_vector[i]; /* TODO: Documentation no 0.5 in documentation! */
  }

  /* f_lunacek_bi_rastrigin_evaluate_core(problem, problem->best_parameter, problem->best_value);
  printf("\n %f , x_opt[0]= %f\n", problem->best_value[0], problem->best_parameter[0]);*/ /* Wassim: for testing purposes, might end up being the one kept though*/

  
  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "block-rotated_weakly-structured");

  coco_random_free(rng);
  coco_free_block_matrix(B1, dimension);
  coco_free_block_matrix(B2, dimension);
  coco_free_memory(P11);
  coco_free_memory(P12);
  coco_free_memory(P21);
  coco_free_memory(P22);
  coco_free_memory(block_sizes1);
  coco_free_memory(block_sizes2);
  coco_free_memory(mu0_vector);
  coco_free_memory(mu1_vector);
  coco_free_memory(sign_vector);
  return problem;
}

