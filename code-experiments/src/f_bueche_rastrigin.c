/**
 * @file f_bueche_rastrigin.c
 * @brief Implementation of the Bueche-Rastrigin function and problem.
 */

#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_vars_brs.c"
#include "transform_vars_oscillate.c"
#include "transform_vars_shift.c"
#include "transform_obj_shift.c"
#include "transform_obj_penalize.c"
#include "transform_obj_norm_by_dim.c"

/**
 * @brief Implements the Bueche-Rastrigin function without connections to any COCO structures.
 */
static double f_bueche_rastrigin_raw(const double *x, const size_t number_of_variables) {

  double tmp = 0., tmp2 = 0.;
  size_t i;
  double result;

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    tmp += cos(2 * coco_pi * x[i]);
    tmp2 += x[i] * x[i];
  }
  result = 10.0 * ((double) (long) number_of_variables - tmp) + tmp2 + 0;
  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_bueche_rastrigin_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_bueche_rastrigin_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Bueche-Rastrigin problem.
 */
static coco_problem_t *f_bueche_rastrigin_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Bueche-Rastrigin function",
      f_bueche_rastrigin_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "bueche-rastrigin", number_of_variables);

  /* Compute best solution */
  f_bueche_rastrigin_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Bueche-Rastrigin problem.
 */
static coco_problem_t *f_bueche_rastrigin_bbob_problem_allocate(const size_t function,
                                                                const size_t dimension,
                                                                const size_t instance,
                                                                const long rseed,
                                                                const char *problem_id_template,
                                                                const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  const double penalty_factor = 100.0;
  size_t i;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  /* OME: This step is in the legacy C code but _not_ in the function description. */
  for (i = 0; i < dimension; i += 2) {
    xopt[i] = fabs(xopt[i]);
  }

  problem = f_bueche_rastrigin_allocate(dimension);
  problem = transform_vars_brs(problem);
  problem = transform_vars_oscillate(problem);
  problem = transform_vars_shift(problem, xopt, 0);

  /*if large scale test-bed, normalize by dim*/
  if (coco_strfind(problem_name_template, "BBOB large-scale suite") >= 0){
    problem = transform_obj_norm_by_dim(problem);
  }
  problem = transform_obj_shift(problem, fopt);
  problem = transform_obj_penalize(problem, penalty_factor);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

  coco_free_memory(xopt);
  return problem;
}
