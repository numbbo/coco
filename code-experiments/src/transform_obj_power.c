/**
 * @file transform_obj_power.c
 * @brief Implementation of raising the objective value to the power of a given exponent.
 */

#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_obj_power.
 */
typedef struct {
  double exponent;
} transform_obj_power_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_obj_power_evaluate(coco_problem_t *problem, const double *x, double *y) {
  transform_obj_power_data_t *data;
  size_t i;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_obj_power_data_t *) coco_problem_transformed_get_data(problem);
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);

  for (i = 0; i < problem->number_of_objectives; i++) {
      y[i] = pow(y[i], data->exponent);
  }
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_obj_power(coco_problem_t *inner_problem, const double exponent) {
  transform_obj_power_data_t *data;
  coco_problem_t *problem;

  data = (transform_obj_power_data_t *) coco_allocate_memory(sizeof(*data));
  data->exponent = exponent;

  problem = coco_problem_transformed_allocate(inner_problem, data, NULL, "transform_obj_power");
  problem->evaluate_function = transform_obj_power_evaluate;
  /* Compute best value */
  transform_obj_power_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
