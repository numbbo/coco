#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "bbob2009_legacy_code.c"

typedef struct {
  int seed;
  double *x;
  coco_free_function_t old_free_problem;
} _x_hat_data_t;

static void _x_hat_evaluate_function(coco_problem_t *self, double *x,
                                     double *y) {
  size_t i;
  _x_hat_data_t *data;
  coco_problem_t *inner_problem;
  data = coco_get_transform_data(self);
  inner_problem = coco_get_transform_inner_problem(self);
  do {
    bbob2009_unif(data->x, self->number_of_variables, data->seed);

    for (i = 0; i < self->number_of_variables; ++i) {
      if (data->x[i] - 0.5 < 0.0) {
        data->x[i] = -x[i];
      } else {
        data->x[i] = x[i];
      }
    }
    coco_evaluate_function(inner_problem, data->x, y);
  } while (0);
}

static void _x_hat_free_data(void *thing) {
  _x_hat_data_t *data = thing;
  coco_free_memory(data->x);
}

/**
 * Multiply the x-vector by the vector 2 * 1+-
 */
coco_problem_t *x_hat(coco_problem_t *inner_problem, int seed) {
  _x_hat_data_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->seed = seed;
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  self =
      coco_allocate_transformed_problem(inner_problem, data, _x_hat_free_data);
  self->evaluate_function = _x_hat_evaluate_function;
  return self;
}
