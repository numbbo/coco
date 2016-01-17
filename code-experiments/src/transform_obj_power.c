#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
  double exponent;
} transform_obj_power_data_t;

static void transform_obj_power_evaluate(coco_problem_t *self, const double *x, double *y) {
  transform_obj_power_data_t *data;
  size_t i;
  data = coco_transformed_get_data(self);
  coco_evaluate_function(coco_transformed_get_inner_problem(self), x, y);
  for (i = 0; i < self->number_of_objectives; i++) {
      y[i] = pow(y[i], data->exponent);
  }
}

/**
 * Raise the objective value to the power of a given exponent.
 */
static coco_problem_t *f_transform_obj_power(coco_problem_t *inner_problem, const double exponent) {
  transform_obj_power_data_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->exponent = exponent;

  self = coco_transformed_allocate(inner_problem, data, NULL);
  self->evaluate_function = transform_obj_power_evaluate;
  /* Compute best value */
  transform_obj_power_evaluate(self, self->best_parameter, self->best_value);
  return self;
}
