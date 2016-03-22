/**
 * @file transform_vars_scale.c
 * @brief Implementation of scaling decision values by a given factor.
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_vars_scale.
 */
typedef struct {
  double factor;
  double *x;
} transform_vars_scale_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_scale_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_scale_data_t *data;
  coco_problem_t *inner_problem;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_scale_data_t *) coco_problem_transformed_get_data(problem);
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

/**
 * @brief Frees the data object.
 */
static void transform_vars_scale_free(void *thing) {
  transform_vars_scale_data_t *data = (transform_vars_scale_data_t *) thing;
  coco_free_memory(data->x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_scale(coco_problem_t *inner_problem, const double factor) {
  transform_vars_scale_data_t *data;
  coco_problem_t *problem;
  size_t i;
  data = (transform_vars_scale_data_t *) coco_allocate_memory(sizeof(*data));
  data->factor = factor;
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_scale_free, "transform_vars_scale");
  problem->evaluate_function = transform_vars_scale_evaluate;
  /* Compute best parameter */
  if (data->factor != 0.) {
      for (i = 0; i < problem->number_of_variables; i++) {
          problem->best_parameter[i] /= data->factor;
      }
  } /* else error? */
  return problem;
}
