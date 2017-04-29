/**
 * @file transform_vars_conditioning.c
 * @brief Implementation of conditioning decision values.
 */

#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_vars_conditioning.
 */
typedef struct {
  double *x;
  double alpha;
} transform_vars_conditioning_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_conditioning_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_conditioning_data_t *data;
  coco_problem_t *inner_problem;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_conditioning_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    /* OME: We could precalculate the scaling coefficients if we
     * really wanted to.
     */
    data->x[i] = pow(data->alpha, 0.5 * (double) (long) i / ((double) (long) problem->number_of_variables - 1.0))
        * x[i];
  }
  coco_evaluate_function(inner_problem, data->x, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Evaluates the gradient of the transformed function.
 */
static void transform_vars_conditioning_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_conditioning_data_t *data;
  coco_problem_t *inner_problem;
  double *gradient;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_conditioning_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  gradient = coco_allocate_vector(inner_problem->number_of_variables);
  
  for (i = 0; i < problem->number_of_variables; ++i) {
    gradient[i] = pow(data->alpha, 0.5 * (double) (long) i / ((double) (long) problem->number_of_variables - 1.0));
    data->x[i] = gradient[i] * x[i];
  }
  coco_evaluate_gradient(inner_problem, data->x, y);
  
  for (i = 0; i < inner_problem->number_of_variables; ++i)
    gradient[i] *= y[i];
  
  for (i = 0; i < inner_problem->number_of_variables; ++i)
    y[i] = gradient[i];
    
  coco_free_memory(gradient);
}

static void transform_vars_conditioning_free(void *thing) {
  transform_vars_conditioning_data_t *data = (transform_vars_conditioning_data_t *) thing;
  coco_free_memory(data->x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_conditioning(coco_problem_t *inner_problem, const double alpha) {
  transform_vars_conditioning_data_t *data;
  coco_problem_t *problem;

  data = (transform_vars_conditioning_data_t *) coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->alpha = alpha;
  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_conditioning_free, "transform_vars_conditioning");
  problem->evaluate_function = transform_vars_conditioning_evaluate;
  problem->evaluate_gradient = transform_vars_conditioning_evaluate_gradient;

  if (coco_problem_best_parameter_not_zero(inner_problem)) {
    coco_warning("transform_vars_conditioning(): 'best_parameter' not updated, set to NAN");
    coco_vector_set_to_nan(inner_problem->best_parameter, inner_problem->number_of_variables);
  }  return problem;
}
