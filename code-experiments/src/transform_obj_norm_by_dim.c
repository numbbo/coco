/**
 * @file transform_obj_norm_by_dim.c
 * @brief Implementation of nomalizing the objective value by dividing the fitness by the number of variables
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_obj_norm_by_dim
 */
typedef struct {
  double offset;
} transform_obj_norm_by_dim_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_obj_norm_by_dim_evaluate(coco_problem_t *problem, const double *x, double *y) {
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  y[0] = y[0] / ((double) problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_obj_norm_by_dim(coco_problem_t *inner_problem) {
  coco_problem_t *problem;
  problem = coco_problem_transformed_allocate(inner_problem, NULL, NULL, "transform_obj_norm_by_dim");
  problem->evaluate_function = transform_obj_norm_by_dim_evaluate;
  transform_obj_norm_by_dim_evaluate(problem, problem->best_parameter, problem->best_value);

  return problem;
}
