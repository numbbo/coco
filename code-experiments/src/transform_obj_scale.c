/**
 * @file transform_obj_factor.c
 * @brief Implementation of multiplying the objective value by a given factor
 * Mostly used to normalize by the dimension in the large-scale suite
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_vars_scale.
 */
typedef struct {
  double factor;
} transform_obj_scale_data_t;


/**
 * @brief Evaluates the transformation.
 */
static void transform_obj_scale_evaluate(coco_problem_t *problem, const double *x, double *y) {
  transform_obj_scale_data_t *data;
  data = (transform_obj_scale_data_t *) coco_problem_transformed_get_data(problem);
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  y[0] *= data->factor;

  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_obj_scale(coco_problem_t *inner_problem, const double factor) {
  transform_obj_scale_data_t *data;
  coco_problem_t *problem;
  
  data = (transform_obj_scale_data_t *) coco_allocate_memory(sizeof(*data));
  data->factor = factor;
  
  problem = coco_problem_transformed_allocate(inner_problem, data, NULL, "transform_obj_scale");
  problem->evaluate_function = transform_obj_scale_evaluate;
  problem->best_value[0] *= factor; /*shouldn't matter as long as fopt = 0 originally*/
  return problem;
}
