/**
 * @file f_sphere.c
 * @brief Implementation of the sphere function and problem.
 */

#include <stdio.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_shift.c"
#include "transform_vars_shift.c"

/**
 * @brief Implements the sphere function without connections to any COCO structures.
 */
static double f_sphere_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double result;
    
  if (coco_vector_contains_nan(x, number_of_variables))
    return NAN;

  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    result += x[i] * x[i];
  }

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_sphere_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_sphere_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Evaluates the gradient of the sphere function.
 */
static void f_sphere_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {

  size_t i;

  for (i = 0; i < problem->number_of_variables; ++i) {
    y[i] = 2.0 * x[i];
  }
}

/**
 * @brief Allocates the basic sphere problem.
 */
static coco_problem_t *f_sphere_allocate(const size_t number_of_variables) {
	
  coco_problem_t *problem = coco_problem_allocate_from_scalars("sphere function",
     f_sphere_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  problem->evaluate_gradient = f_sphere_evaluate_gradient;
  coco_problem_set_id(problem, "%s_d%02lu", "sphere", number_of_variables);

  /* Compute best solution */
  f_sphere_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB sphere problem.
 */
static coco_problem_t *f_sphere_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const long rseed,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {

  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  fopt = bbob2009_compute_fopt(function, instance);

  problem = f_sphere_allocate(dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

  coco_free_memory(xopt);
  return problem;
}

