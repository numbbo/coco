#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob2009_legacy_code.c"
#include "transform_vars_brs.c"
#include "transform_vars_oscillate.c"
#include "transform_vars_shift.c"
#include "transform_obj_shift.c"
#include "transform_obj_penalize.c"

static double f_bueche_rastrigin_raw(const double *x, const size_t number_of_variables) {

  double tmp = 0., tmp2 = 0.;
  size_t i;
  double result;

  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    tmp += cos(2 * coco_pi * x[i]);
    tmp2 += x[i] * x[i];
  }
  result = 10.0 * ((double) (long) number_of_variables - tmp) + tmp2 + 0;
  return result;
}

static void f_bueche_rastrigin_evaluate(coco_problem_t *self, const double *x, double *y) {
  assert(self->number_of_objectives == 1);
  y[0] = f_bueche_rastrigin_raw(x, self->number_of_variables);
}

static coco_problem_t *f_bueche_rastrigin_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Bueche-Rastrigin function",
      f_bueche_rastrigin_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%04lu", "bueche-rastrigin", number_of_variables);

  /* Compute best solution */
  f_bueche_rastrigin_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

static coco_problem_t *f_bueche_rastrigin_bbob_problem_allocate(const size_t function_id,
                                                                const size_t dimension,
                                                                const size_t instance_id,
                                                                const long rseed,
                                                                const char *problem_id_template,
                                                                const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  const double penalty_factor = 100.0;
  size_t i;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function_id, instance_id);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  /* OME: This step is in the legacy C code but _not_ in the function description. */
  for (i = 0; i < dimension; i += 2) {
    xopt[i] = fabs(xopt[i]);
  }

  problem = f_bueche_rastrigin_allocate(dimension);
  problem = f_transform_vars_brs(problem);
  problem = f_transform_vars_oscillate(problem);
  problem = f_transform_vars_shift(problem, xopt, 0);
  problem = f_transform_obj_shift(problem, fopt);
  problem = f_transform_obj_penalize(problem, penalty_factor);

  coco_problem_set_id(problem, problem_id_template, function_id, instance_id, dimension);
  coco_problem_set_name(problem, problem_name_template, function_id, instance_id, dimension);

  coco_free_memory(xopt);
  return problem;
}


/* TODO: Deprecated functions below are to be deleted when the new ones work as they should */

static void deprecated__f_bueche_rastrigin_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  double tmp = 0., tmp2 = 0.;
  assert(self->number_of_objectives == 1);
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    tmp += cos(2 * coco_pi * x[i]);
    tmp2 += x[i] * x[i];
  }
  y[0] = 10.0 * ((double) (long) self->number_of_variables - tmp) + tmp2 + 0;
}

static coco_problem_t *deprecated__f_bueche_rastrigin(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_problem_allocate(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("Bueche-Rastrigin function");
  /* Construct a meaningful problem id */
  problem_id_length = (size_t) snprintf(NULL, 0, "%s_%02lu", "bueche-rastrigin", (long) number_of_variables);
  problem->problem_id = (char *) coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02lu", "skewRastriginBueche",
      (long) number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = deprecated__f_bueche_rastrigin_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }
  /* Calculate best parameter value */
  deprecated__f_bueche_rastrigin_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
