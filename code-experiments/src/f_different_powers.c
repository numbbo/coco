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
 * @brief Implements the sign function.
 */
double sign(double x) {
  
  if (x > 0) return 1;
  if (x < 0) return -1;
  return 0;
}

/**
 * @brief Evaluates the gradient of the function "different powers".
 */
static void f_different_powers_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {

  size_t i;
  double sum = 0.0;
  double aux;

  for (i = 0; i < problem->number_of_variables; ++i) {
    aux = 2.0 + (4.0 * (double) (long) i) / ((double) (long) problem->number_of_variables - 1.0);
    sum += pow(fabs(x[i]), aux);
  }
  
  for (i = 0; i < problem->number_of_variables; ++i) {
    aux = 2.0 + (4.0 * (double) (long) i) / ((double) (long) problem->number_of_variables - 1.0);
	 y[i] = 0.5 * (aux)/(sum);
    aux -= 1.0;
    y[i] *= pow(fabs(x[i]), aux) * sign(x[i]);
  }
  
}

/**
 * @brief Allocates the basic different powers problem.
 */
static coco_problem_t *f_different_powers_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("different powers function",
      f_different_powers_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  problem->evaluate_gradient = f_different_powers_evaluate_gradient;
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
