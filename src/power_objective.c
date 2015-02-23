#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct { double exponent; } _powo_data_t;

static void _powo_evaluate_function(coco_problem_t *self, double *x,
                                    double *y) {
  _powo_data_t *data;
  data = coco_get_transform_data(self);
  coco_evaluate_function(coco_get_transform_inner_problem(self), x, y);
  y[0] = pow(y[0], data->exponent);
}

/**
 * Raise the objective value to the power of a given exponent.
 */
coco_problem_t *power_objective(coco_problem_t *inner_problem,
                                const double exponent) {
  _powo_data_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->exponent = exponent;

  self = coco_allocate_transformed_problem(inner_problem, data, NULL);
  self->evaluate_function = _powo_evaluate_function;
  return self;
}
