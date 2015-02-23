#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct { double offset; } _shift_objective_t;

static void _so_evaluate_function(coco_problem_t *self, double *x, double *y) {
  _shift_objective_t *data;
  data = coco_get_transform_data(self);
  coco_evaluate_function(coco_get_transform_inner_problem(self), x, y);
  y[0] += data->offset;
}

/**
 * Shift the objective value of the inner problem by amount.
 */
coco_problem_t *shift_objective(coco_problem_t *inner_problem,
                                const double offset) {
  coco_problem_t *self;
  _shift_objective_t *data;
  data = coco_allocate_memory(sizeof(*data));
  data->offset = offset;

  self = coco_allocate_transformed_problem(inner_problem, data, NULL);
  self->evaluate_function = _so_evaluate_function;
  self->best_value[0] += offset;
  return self;
}
