#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
  double offset;
} transform_obj_shift_data_t;

static void transform_obj_shift_evaluate(coco_problem_t *self, const double *x, double *y) {
  transform_obj_shift_data_t *data;
  data = coco_transformed_get_data(self);
  coco_evaluate_function(coco_transformed_get_inner_problem(self), x, y);
  y[0] += data->offset; /* FIXME: shifts only the first objective */
}

/**
 * Shift the objective value of the inner problem by offset.
 */
static coco_problem_t *f_transform_obj_shift(coco_problem_t *inner_problem, const double offset) {
  coco_problem_t *self;
  transform_obj_shift_data_t *data;
  data = coco_allocate_memory(sizeof(*data));
  data->offset = offset;

  self = coco_transformed_allocate(inner_problem, data, NULL);
  self->evaluate_function = transform_obj_shift_evaluate;
  self->best_value[0] += offset; /* FIXME: shifts only the first objective */
  return self;
}
