#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
  double offset;
} transform_obj_shift_data_t;

static void transform_obj_shift_evaluate(coco_problem_t *problem, const double *x, double *y) {
  transform_obj_shift_data_t *data;
  size_t i;
  data = coco_problem_transformed_get_data(problem);
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  for (i = 0; i < problem->number_of_objectives; i++) {
      y[i] += data->offset;
  }
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * Shift the objective value of the inner problem by offset.
 */
static coco_problem_t *f_transform_obj_shift(coco_problem_t *inner_problem, const double offset) {
  coco_problem_t *problem;
  transform_obj_shift_data_t *data;
  size_t i;
  data = coco_allocate_memory(sizeof(*data));
  data->offset = offset;

  problem = coco_problem_transformed_allocate(inner_problem, data, NULL);
  problem->evaluate_function = transform_obj_shift_evaluate;
  for (i = 0; i < problem->number_of_objectives; i++) {
      problem->best_value[0] += offset;
  }
  return problem;
}
