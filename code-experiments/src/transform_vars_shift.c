#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
  double *offset;
  double *shifted_x;
  coco_free_function_t old_free_problem;
} transform_vars_shift_data_t;

static void transform_vars_shift_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  transform_vars_shift_data_t *data;
  coco_problem_t *inner_problem;

  data = coco_transformed_get_data(self);
  inner_problem = coco_transformed_get_inner_problem(self);

  for (i = 0; i < self->number_of_variables; ++i) {
    data->shifted_x[i] = x[i] - data->offset[i];
  }
  coco_evaluate_function(inner_problem, data->shifted_x, y);
  assert(y[0] >= self->best_value[0]);
}

static void transform_vars_shift_free(void *thing) {
  transform_vars_shift_data_t *data = thing;
  coco_free_memory(data->shifted_x);
  coco_free_memory(data->offset);
}

/*
 * Shift all variables of ${inner_problem} by ${offset}.
 */
static coco_problem_t *f_transform_vars_shift(coco_problem_t *inner_problem,
                                              const double *offset,
                                              const int shift_bounds) {
  transform_vars_shift_data_t *data;
  coco_problem_t *self;
  if (shift_bounds)
    coco_error("shift_bounds not implemented.");

  data = coco_allocate_memory(sizeof(*data));
  data->offset = coco_duplicate_vector(offset, inner_problem->number_of_variables);
  data->shifted_x = coco_allocate_vector(inner_problem->number_of_variables);

  self = coco_transformed_allocate(inner_problem, data, transform_vars_shift_free);
  self->evaluate_function = transform_vars_shift_evaluate;
  return self;
}
