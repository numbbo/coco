/**
 * @file transform_obj_scale.c
 * @brief Implementation of scaling the objective value by the given factor.
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_obj_scale.
 */
typedef struct {
  double factor;
} transform_obj_scale_data_t;

/**
 * @brief Evaluates the transformed function.
 */
static void transform_obj_scale_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  transform_obj_scale_data_t *data;
  double *cons_values;
  int is_feasible;
  size_t i;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
    coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
    return;
  }

  data = (transform_obj_scale_data_t *) coco_problem_transformed_get_data(problem);
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);

  for (i = 0; i < problem->number_of_objectives; i++)
    y[i] *= data->factor;

  if (problem->number_of_constraints > 0) {
    cons_values = coco_allocate_vector(problem->number_of_constraints);
    is_feasible = coco_is_feasible(problem, x, cons_values);
    coco_free_memory(cons_values);
    if (is_feasible)
      assert(y[0] + 1e-13 >= problem->best_value[0]);
  }
  else assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Evaluates the gradient of the transformed function at x
 */
static void transform_obj_scale_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_obj_scale_data_t *data;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
    coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
    return;
  }

  bbob_evaluate_gradient(coco_problem_transformed_get_inner_problem(problem), x, y);

  data = (transform_obj_scale_data_t *) coco_problem_transformed_get_data(problem);
  for (i = 0; i < problem->number_of_variables; ++i) {
    y[i] *= data->factor;
  }
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_obj_scale(coco_problem_t *inner_problem, const double factor) {
  coco_problem_t *problem;
  transform_obj_scale_data_t *data;
  size_t i;
  data = (transform_obj_scale_data_t *) coco_allocate_memory(sizeof(*data));
  data->factor = factor;

  problem = coco_problem_transformed_allocate(inner_problem, data,
    NULL, "transform_obj_scale");

  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_obj_scale_evaluate_function;

  problem->evaluate_gradient = transform_obj_scale_evaluate_gradient;

  for (i = 0; i < problem->number_of_objectives; ++i)
    problem->best_value[i] *= factor;

  return problem;
}
