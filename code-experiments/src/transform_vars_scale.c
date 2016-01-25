/*
 * Scale variables by a given factor.
 */
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
  double factor;
  double *x;
} transform_vars_scale_data_t;

static void transform_vars_scale_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_scale_data_t *data;
  coco_problem_t *inner_problem;
  data = coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);
  do {
    const double factor = data->factor;

    for (i = 0; i < problem->number_of_variables; ++i) {
      data->x[i] = factor * x[i];
    }
    coco_evaluate_function(inner_problem, data->x, y);
    assert(y[0] + 1e-13 >= problem->best_value[0]);
  } while (0);
}

static void transform_vars_scale_free(void *thing) {
  transform_vars_scale_data_t *data = thing;
  coco_free_memory(data->x);
}

/**
 * Scale all variables by factor before evaluation.
 */
static coco_problem_t *f_transform_vars_scale(coco_problem_t *inner_problem, const double factor) {
  transform_vars_scale_data_t *data;
  coco_problem_t *problem;

  data = coco_allocate_memory(sizeof(*data));
  data->factor = factor;
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_scale_free);
  problem->evaluate_function = transform_vars_scale_evaluate;
  return problem;
}
