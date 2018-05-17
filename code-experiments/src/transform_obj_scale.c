/**
 * @file transform_obj_scale.c
 * @brief Scale the objective value(s) by some given factor.
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
 * @brief Evaluate the transformation, scales the first objective value only.
 *
 * This function is used for the large-scale suite.
 */
static void transform_obj_scale_evaluate(coco_problem_t *problem, const double *x, double *y) {
  transform_obj_scale_data_t *data;

  data = (transform_obj_scale_data_t *) coco_problem_transformed_get_data(problem);
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  y[0] *= data->factor;

  /* This is too expensive and non-removable for an assertion (NH)
  if (coco_is_feasible(problem, x, NULL))
  */
  if (coco_problem_get_number_of_constraints(problem) <= 0)
    assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Evaluates the transformed function, scales all objective values.
 *
 * This function is used in the constrained case.
 */
static void transform_obj_scale_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  transform_obj_scale_data_t *data;
  double *cons_values;
  size_t i;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
    coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
    return;
  }

  data = (transform_obj_scale_data_t *) coco_problem_transformed_get_data(problem);
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);

  for (i = 0; i < problem->number_of_objectives; i++)
    y[i] *= data->factor;

  /* This is too expensive and non-removable for an assertion (NH)
  if (coco_is_feasible(problem, x, NULL))
  */
  if (coco_problem_get_number_of_constraints(problem) <= 0)
    assert(y[0] + 1e-13 >= problem->best_value[0]);
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
  
  problem = coco_problem_transformed_allocate(inner_problem, data, NULL, "transform_obj_scale");

  if (inner_problem->number_of_constraints <= 0 && inner_problem->number_of_objectives == 1) {
    problem->evaluate_function = transform_obj_scale_evaluate;
    problem->best_value[0] *= factor;
  }
  else if inner_problem->number_of_objectives >= 1 {
    problem->evaluate_function = transform_obj_scale_evaluate_function; /* handles constraints adequately */
    problem->evaluate_gradient = transform_obj_scale_evaluate_gradient;
    for (i = 0; i < problem->number_of_objectives; ++i)
      problem->best_value[i] *= factor;
  } else
    coco_error("transform_obj_scale called with < 1 objectives.");

  return problem;
}
