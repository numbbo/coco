/*
 * Scale variables by a given factor.
 */
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
  double factor;
  double *x;
} _scv_data_t;

static void _scv_evaluate_function(coco_problem_t *self, double *x, double *y) {
  size_t i;
  _scv_data_t *data;
  coco_problem_t *inner_problem;
  data = coco_get_transform_data(self);
  inner_problem = coco_get_transform_inner_problem(self);
  do {
    const double factor = data->factor;

    for (i = 0; i < self->number_of_variables; ++i) {
      data->x[i] = factor * x[i];
    }
    coco_evaluate_function(inner_problem, data->x, y);
  } while (0);
}

static void _scv_free_data(void *thing) {
  _scv_data_t *data = thing;
  coco_free_memory(data->x);
}

/**
 * Scale all variables by factor before evaluation.
 */
coco_problem_t *scale_variables(coco_problem_t *inner_problem,
                                const double factor) {
  _scv_data_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->factor = factor;
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  self = coco_allocate_transformed_problem(inner_problem, data, _scv_free_data);
  self->evaluate_function = _scv_evaluate_function;
  return self;
}
