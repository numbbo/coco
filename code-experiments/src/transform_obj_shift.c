/**
 * @file transform_obj_shift.c
 * @brief Implementation of shifting the objective value by the given offset.
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_obj_shift.
 */
typedef struct {
  double offset;
} transform_obj_shift_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_obj_shift_evaluate(coco_problem_t *problem, const double *x, double *y) {
  transform_obj_shift_data_t *data;
  size_t i;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_obj_shift_data_t *) coco_problem_transformed_get_data(problem);
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);

  for (i = 0; i < problem->number_of_objectives; i++) {
    y[i] += data->offset;
  }
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_obj_shift(coco_problem_t *inner_problem, const double offset) {
  coco_problem_t *problem;
  transform_obj_shift_data_t *data;
  size_t i;
  data = (transform_obj_shift_data_t *) coco_allocate_memory(sizeof(*data));
  data->offset = offset;

  problem = coco_problem_transformed_allocate(inner_problem, data, NULL, "transform_obj_shift");
  problem->evaluate_function = transform_obj_shift_evaluate;
  for (i = 0; i < problem->number_of_objectives; i++) {
      problem->best_value[0] += offset;
  }
  return problem;
}
