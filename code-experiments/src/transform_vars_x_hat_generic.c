/**
 * @file transform_vars_x_hat_generic.c
 * @brief Implementation of multiplying the decision values by the vector 1+-.
 * Wassim: TODO: should eventually replace the non generic version in its use in Schwefel where xopt would be set elsewhere
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"

/**
 * @brief Data type for transform_vars_x_hat_generic.
 */
typedef struct {
  double *sign_vector;
  double *x;
  coco_problem_free_function_t old_free_problem;
} transform_vars_x_hat_generic_data_t;

/**
 * @brief Data type for the versatile_data_t
 */
typedef struct {
  coco_problem_t *sub_problem_mu0;
  coco_problem_t *sub_problem_mu1;
  double *x_hat;
} f_lunacek_bi_rastrigin_versatile_data_t;


/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_x_hat_generic_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_x_hat_generic_data_t *data;
  coco_problem_t *inner_problem;
  data = (transform_vars_x_hat_generic_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    data->x[i] = 2.0 * data->sign_vector[i] * x[i];
    ((f_lunacek_bi_rastrigin_versatile_data_t *) problem->versatile_data)->x_hat[i] = data->x[i];
  }
  coco_evaluate_function(inner_problem, data->x, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_x_hat_generic_free(void *thing) {
  transform_vars_x_hat_generic_data_t *data = (transform_vars_x_hat_generic_data_t *) thing;
  coco_free_memory(data->x);
  coco_free_memory(data->sign_vector);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_x_hat_generic(coco_problem_t *inner_problem, const double *sign_vector) {
  transform_vars_x_hat_generic_data_t *data;
  coco_problem_t *problem;
  size_t i;

  data = (transform_vars_x_hat_generic_data_t *) coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->sign_vector = coco_allocate_vector(inner_problem->number_of_variables);
  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    data->sign_vector[i] = sign_vector[i];
  }

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_x_hat_generic_free, "transform_vars_x_hat_generic");
  problem->evaluate_function = transform_vars_x_hat_generic_evaluate;

  return problem;
}


