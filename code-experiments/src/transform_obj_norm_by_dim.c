/**
 * @file transform_obj_norm_by_dim.c
 * @brief Implementation of normalizing the raw fitness functions by the dimensions
 * Mostly used to in the large-scale testsuite
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Evaluates the transformation.
 */
static void transform_obj_norm_by_dim_evaluate(coco_problem_t *problem, const double *x, double *y) {

  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  y[0] *= bbob2009_fmin(1, 40. / ((double) problem->number_of_variables));
  /* Wassim: might want to use a function (with no 40) here that we can put in a helpers file */

  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_obj_norm_by_dim(coco_problem_t *inner_problem) {
  coco_problem_t *problem;

  problem = coco_problem_transformed_allocate(inner_problem, NULL, NULL, "transform_obj_norm_by_dim");
  problem->evaluate_function = transform_obj_norm_by_dim_evaluate;
  /*problem->best_value[0] *= 1;*/ /*shouldn't matter*/
  return problem;
}
