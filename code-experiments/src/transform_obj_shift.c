#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
  double offset;
} transform_obj_shift_data_t;

static void transform_obj_shift_evaluate(coco_problem_t *self, const double *x, double *y) {
  transform_obj_shift_data_t *data;
  size_t i;
  data = coco_transformed_get_data(self);
  coco_evaluate_function(coco_transformed_get_inner_problem(self), x, y);
  for (i = 0; i < self->number_of_objectives; i++) {
      y[i] += data->offset;
  }
}

/**
 * Shift the objective value of the inner problem by offset.
 */
static coco_problem_t *f_transform_obj_shift(coco_problem_t *inner_problem, const double offset) {
  coco_problem_t *self;
  transform_obj_shift_data_t *data;
  size_t i;
  data = coco_allocate_memory(sizeof(*data));
  data->offset = offset;

  self = coco_transformed_allocate(inner_problem, data, NULL);
  self->evaluate_function = transform_obj_shift_evaluate;
  for (i = 0; i < self->number_of_objectives; i++) {
      self->best_value[0] += offset;
  }
  return self;
}
