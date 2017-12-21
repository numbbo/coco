/**
 * @file transform_vars_round_step.c
 * @brief Implementation of rounding the varaibles for the step-ellispoid function
 * TODO: should this be a helper function instead?
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "coco_utilities.c"

/**
 * @brief Data type for transform_vars_round_step.
 */
typedef struct {
  double alpha;
  double *rounded_x;
} transform_vars_round_step_data_t;

/**
 * @brief Data type to be used in problem->versatile_data
 */
typedef struct {
  double zhat_1; /**< @brief contains the value of \hat{z}_1 that is used to compute the fintess */ 
} f_step_ellipsoid_versatile_data_t;


/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_round_step_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_round_step_data_t *data;
  coco_problem_t *inner_problem;
  
  data = (transform_vars_round_step_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);
  /* multiplication by d to counter-balance the normalization by d*/
  ((f_step_ellipsoid_versatile_data_t *) problem->versatile_data)->zhat_1 = fabs(x[0]) * (double) inner_problem->number_of_variables;/* TODO: Discuss: consider not pre-imptively multiplying by dim to not change the outcome of the max in the core function even though we might want to keep it as it is since otherwise, the sum part of the max may take over as dim increases */
  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    if (fabs(x[i]) > 0.5){
      data->rounded_x[i] = coco_double_round(x[i]);
    } else {
      data->rounded_x[i] = coco_double_round(data->alpha * x[i]) / data->alpha;
    }
  }
  coco_evaluate_function(inner_problem, data->rounded_x, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_round_step_free(void *thing) {
  transform_vars_round_step_data_t *data = (transform_vars_round_step_data_t *) thing;
  coco_free_memory(data->rounded_x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_round_step(coco_problem_t *inner_problem, const double alpha) {
  transform_vars_round_step_data_t *data;
  coco_problem_t *problem;
  size_t i;
  
  data = (transform_vars_round_step_data_t *) coco_allocate_memory(sizeof(*data));
  data->rounded_x = coco_allocate_vector(inner_problem->number_of_variables + 1);
  data->alpha = alpha;
  
  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_round_step_free, "transform_vars_round_step");
  problem->evaluate_function = transform_vars_round_step_evaluate;
  /* Compute best parameter */
  for (i = 0; i < problem->number_of_variables; i++) {
    if (fabs(problem->best_parameter[i]) > 0.5) {
      problem->best_parameter[i] = coco_double_round(problem->best_parameter[i]);
    } else {
      problem->best_parameter[i] = coco_double_round(data->alpha * problem->best_parameter[i]) / data->alpha;
    }
  }
  return problem;
}
