/**
 * @file mi_f_sphere.c
 * @brief Implementation of the mixed-integer sphere function and problem.
 */

#include <stdio.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_shift.c"
#include "transform_vars_shift.c"
#include "f_sphere.c"

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 *
 * @note Takes care of rounding x
 */
static void mi_f_sphere_evaluate(coco_problem_t *problem, const double *x, double *y) {
  double *x_rounded;
  assert(problem->number_of_objectives == 1);
  x_rounded = coco_duplicate_vector(x, problem->number_of_variables);
  y[0] = f_sphere_raw(x_rounded, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
  coco_free_memory(x_rounded);
}

/**
 * @brief Evaluates the gradient of the sphere function.
 *
 * @note Takes care of rounding x
 */
static void mi_f_sphere_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {

  size_t i;
  double *x_rounded;

  x_rounded = coco_duplicate_vector(x, problem->number_of_variables);
  for (i = 0; i < problem->number_of_variables; ++i) {
    y[i] = 2.0 * x_rounded[i];
  }
  coco_free_memory(x_rounded);
}

/**
 * @brief Allocates the mixed-integer sphere problem.
 */
static coco_problem_t *mi_f_sphere_allocate(const size_t number_of_variables,
                                            const double *smallest_values_of_interest,
                                            const double *largest_values_of_interest,
                                            const int *are_variables_integer) {

  double *best_parameter = coco_allocate_vector_with_value(number_of_variables, 0.0);
  coco_problem_t *problem = coco_problem_allocate_from_arrays("mixed-integer sphere function",
     mi_f_sphere_evaluate, NULL, number_of_variables, 0, smallest_values_of_interest,
     largest_values_of_interest, are_variables_integer, best_parameter);
  problem->evaluate_gradient = mi_f_sphere_evaluate_gradient;
  coco_problem_set_id(problem, "%s_d%02lu", "mi-sphere", number_of_variables);

  /* Compute best solution */
  mi_f_sphere_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB mixed-integer sphere problem.
 */
static coco_problem_t *mi_f_sphere_bbob_problem_allocate(const size_t function,
                                                         const size_t dimension,
                                                         const size_t instance,
                                                         const long rseed,
                                                         const char *problem_id_template,
                                                         const char *problem_name_template,
                                                         const double *smallest_values_of_interest,
                                                         const double *largest_values_of_interest,
                                                         const int *are_variables_integer) {

  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  fopt = bbob2009_compute_fopt(function, instance);

  problem = mi_f_sphere_allocate(dimension, smallest_values_of_interest,
      largest_values_of_interest, are_variables_integer);
  coco_problem_round_solution(problem, xopt);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

  coco_free_memory(xopt);
  return problem;
}

