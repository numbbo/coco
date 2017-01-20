/**
 * @file f_schwefel_generalized.c
 * @brief Implementation of the Schwefel function and problem.
 */

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_vars_conditioning.c"
#include "transform_obj_shift.c"
#include "transform_vars_scale.c"
#include "transform_vars_affine.c"
#include "transform_vars_shift.c"
#include "transform_vars_z_hat.c"
#include "transform_vars_x_hat.c"
#include "transform_obj_norm_by_dim.c"

/**
 * @brief Implements the Schwefel function for large scale.
 */
static double f_schwefel_generalized_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double result;
  double penalty, sum;
  const double Schwefel_constant = 4.189828872724339;

  /* Boundary handling*/
  penalty = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    const double tmp = fabs(x[i]) - 500.0;
    if (tmp > 0.0)
      penalty += tmp * tmp;
  }

  /* Computation core */
  sum = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    sum += x[i] * sin(sqrt(fabs(x[i])));
  }
  result = 0.01 * (penalty + Schwefel_constant*100. - sum / (double) number_of_variables);
  /*result = 0.01 * (penalty - sum);*/

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_schwefel_generalized_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_schwefel_generalized_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Schwefel problem.
 */
static coco_problem_t *f_schwefel_generalized_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Schwefel function",
      f_schwefel_generalized_evaluate, NULL, number_of_variables, -5.0, 5.0, 420.96874633);
  coco_problem_set_id(problem, "%s_d%02lu", "schwefel", number_of_variables);

  /* Compute best solution: best_parameter[i] = 200 * fabs(xopt[i]) */
  f_schwefel_generalized_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Schwefel problem.
 */
static coco_problem_t *f_schwefel_generalized_bbob_problem_allocate(const size_t function,
                                                        const size_t dimension,
                                                        const size_t instance,
                                                        const long rseed,
                                                        const char *problem_id_template,
                                                        const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  size_t i;
  /*const double Schwefel_constant = 4.189828872724339;*/ /* Wassim: now inside the raw function*/
  double *tmp1 = coco_allocate_vector(dimension);
  double *tmp2 = coco_allocate_vector(dimension);

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_unif(tmp1, dimension, rseed);
  for (i = 0; i < dimension; ++i) {
    xopt[i] = 0.5 * 4.2096874637;
    if (tmp1[i] - 0.5 < 0) {
      xopt[i] *= -1;
    }
  }

  for (i = 0; i < dimension; ++i) {
    tmp1[i] = -2 * fabs(xopt[i]);
    tmp2[i] = 2 * fabs(xopt[i]);
  }

  problem = f_schwefel_generalized_allocate(dimension);
  problem = transform_vars_scale(problem, 100);
  problem = transform_vars_shift(problem, tmp1, 0);
  problem = transform_vars_conditioning(problem, 10.0);
  problem = transform_vars_shift(problem, tmp2, 0);
  problem = transform_vars_z_hat(problem, xopt);
  problem = transform_vars_scale(problem, 2);
  problem = transform_vars_x_hat(problem, rseed);
  /*problem = transform_obj_norm_by_dim(problem);*/ /* Wassim: there is already a normalization by dimension*/
  /*problem = transform_obj_shift(problem, Schwefel_constant);*/ /* Wassim: now in the raw definition */
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "large_scale_block_rotated");/*TODO: no large scale prefix*/

  coco_free_memory(tmp1);
  coco_free_memory(tmp2);
  coco_free_memory(xopt);
  return problem;
}
