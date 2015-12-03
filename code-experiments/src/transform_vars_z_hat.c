#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
  double *xopt;
  double *z;
  coco_free_function_t old_free_problem;
} transform_vars_z_hat_data_t;

static void transform_vars_z_hat_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  transform_vars_z_hat_data_t *data;
  coco_problem_t *inner_problem;

  data = coco_transformed_get_data(self);
  inner_problem = coco_transformed_get_inner_problem(self);

  data->z[0] = x[0];

  for (i = 1; i < self->number_of_variables; ++i) {
    data->z[i] = x[i] + 0.25 * (x[i - 1] - 2.0 * fabs(data->xopt[i - 1]));
  }
  coco_evaluate_function(inner_problem, data->z, y);
  assert(y[0] >= self->best_value[0]);
}

static void transform_vars_z_hat_free(void *thing) {
  transform_vars_z_hat_data_t *data = thing;
  coco_free_memory(data->xopt);
  coco_free_memory(data->z);
}

/*
 * Compute the vector {z^hat} for the BBOB Schwefel function.
 */
static coco_problem_t *f_transform_vars_z_hat(coco_problem_t *inner_problem, const double *xopt) {
  transform_vars_z_hat_data_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->xopt = coco_duplicate_vector(xopt, inner_problem->number_of_variables);
  data->z = coco_allocate_vector(inner_problem->number_of_variables);

  self = coco_transformed_allocate(inner_problem, data, transform_vars_z_hat_free);
  self->evaluate_function = transform_vars_z_hat_evaluate;
  return self;
}
