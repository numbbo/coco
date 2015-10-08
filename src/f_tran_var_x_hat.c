#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob2009_legacy_code.c"

typedef struct {
  long seed;
  double *x;
  coco_free_function_t old_free_problem;
} _tv_xh_data_t;

static void private_evaluate_function_tv_xh(coco_problem_t *self, const double *x,
                                     double *y) {
  size_t i;
  _tv_xh_data_t *data;
  coco_problem_t *inner_problem;
  data = coco_get_transform_data(self);
  inner_problem = coco_get_transform_inner_problem(self);
  do {
    bbob2009_unif(data->x, (long)self->number_of_variables, data->seed);

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

static void private_free_data_tv_xh(void *thing) {
  _tv_xh_data_t *data = thing;
  coco_free_memory(data->x);
}

/**
 * Multiply the x-vector by the vector 2 * 1+-.
 */
static coco_problem_t *f_tran_var_x_hat(coco_problem_t *inner_problem, long seed) {
  _tv_xh_data_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->seed = seed;
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  self =
      coco_allocate_transformed_problem(inner_problem, data, private_free_data_tv_xh);
  self->evaluate_function = private_evaluate_function_tv_xh;
  return self;
}
