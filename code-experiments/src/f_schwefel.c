#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "coco_generics.c"
#include "suite_bbob2009_legacy_code.c"
#include "transform_obj_shift.c"
#include "transform_vars_scale.c"
#include "transform_vars_affine.c"
#include "transform_vars_shift.c"
#include "transform_vars_z_hat.c"
#include "transform_vars_x_hat.c"

static double f_schwefel_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double result;
  double penalty, sum;

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
  result = 0.01 * (penalty + 418.9828872724339 - sum / (double) number_of_variables);

  return result;
}

static void f_schwefel_evaluate(coco_problem_t *self, const double *x, double *y) {
  assert(self->number_of_objectives == 1);
  y[0] = f_schwefel_raw(x, self->number_of_variables);
  assert(y[0] >= self->best_value[0]);
}

static coco_problem_t *f_schwefel_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Schwefel function",
      f_schwefel_evaluate, NULL, number_of_variables, -5.0, 5.0, NAN);
  coco_problem_set_id(problem, "%s_d%04lu", "schwefel", number_of_variables);

  /* Compute best solution
   *
   * OME: Hard code optimal value for now...
   * TODO: best_parameter is known, it needs to be saved instead of NAN!!!
   */
  problem->best_value[0] = 0.0;
  return problem;
}

static coco_problem_t *f_schwefel_bbob_problem_allocate(const size_t function,
                                                        const size_t dimension,
                                                        const size_t instance,
                                                        const long rseed,
                                                        const char *problem_id_template,
                                                        const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  size_t i, j;

  const double condition = 10.;

  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *current_row;

  double *tmp1 = coco_allocate_vector(dimension);
  double *tmp2 = coco_allocate_vector(dimension);

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_unif(tmp1, dimension, rseed);
  for (i = 0; i < dimension; ++i) {
    xopt[i] = 0.5 * 4.2096874633;
    if (tmp1[i] - 0.5 < 0) {
      xopt[i] *= -1;
    }
  }

  for (i = 0; i < dimension; ++i) {
    b[i] = 0.0;
    current_row = M + i * dimension;
    for (j = 0; j < dimension; ++j) {
      current_row[j] = 0.0;
      if (i == j) {
        double exponent = 1.0 * (int) i / ((double) (long) dimension - 1);
        current_row[j] = pow(sqrt(condition), exponent);
      }
    }
  }

  for (i = 0; i < dimension; ++i) {
    tmp1[i] = -2 * fabs(xopt[i]);
    tmp2[i] = 2 * fabs(xopt[i]);
  }

  problem = f_schwefel_allocate(dimension);
  problem = f_transform_obj_shift(problem, fopt);
  problem = f_transform_vars_scale(problem, 100);
  problem = f_transform_vars_shift(problem, tmp1, 0);
  problem = f_transform_vars_affine(problem, M, b, dimension);
  problem = f_transform_vars_shift(problem, tmp2, 0);
  problem = f_transform_vars_z_hat(problem, xopt);
  problem = f_transform_vars_scale(problem, 2);
  problem = f_transform_vars_x_hat(problem, rseed);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "5-weakly-structured");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(tmp1);
  coco_free_memory(tmp2);
  coco_free_memory(xopt);
  return problem;
}

/* TODO: Deprecated functions below are to be deleted when the new ones work as they should */

static void deprecated__f_schwefel_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  double penalty, sum;
  assert(self->number_of_objectives == 1);

  /* Boundary handling*/
  penalty = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    const double tmp = fabs(x[i]) - 500.0;
    if (tmp > 0.0)
      penalty += tmp * tmp;
  }

  /* Computation core */
  sum = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    sum += x[i] * sin(sqrt(fabs(x[i])));
  }
  y[0] = 0.01 * (penalty + 418.9828872724339 - sum / (double) self->number_of_variables);
  assert(y[0] >= self->best_value[0]);
}

static coco_problem_t *deprecated__f_schwefel(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_problem_allocate(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("schwefel function");
  /* Construct a meaningful problem id */
  problem_id_length = (size_t) snprintf(NULL, 0, "%s_%02lu", "schwefel", (long) number_of_variables);
  problem->problem_id = (char *) coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02lu", "schwefel", (long) number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = deprecated__f_schwefel_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = NAN;
  }
  /* "Calculate" best parameter value
   *
   * OME: Hard code optimal value for now...
   */
  problem->best_value[0] = 0.0;

  return problem;
}
