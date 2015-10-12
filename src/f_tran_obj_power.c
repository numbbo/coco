#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
  double exponent;
} _tran_obj_power_data_t;

static void private_tran_obj_power_evaluate(coco_problem_t *self, const double *x, double *y) {
  _tran_obj_power_data_t *data;
  data = coco_get_transform_data(self);
  coco_evaluate_function(coco_get_transform_inner_problem(self), x, y);
  y[0] = pow(y[0], data->exponent);
}

/**
 * Raise the objective value to the power of a given exponent.
 */
static coco_problem_t *f_tran_obj_power(coco_problem_t *inner_problem, const double exponent) {
  _tran_obj_power_data_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->exponent = exponent;

  self = coco_allocate_transformed_problem(inner_problem, data, NULL);
  self->evaluate_function = private_tran_obj_power_evaluate;
  return self;
}
