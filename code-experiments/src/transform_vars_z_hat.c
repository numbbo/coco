/**
 * @file transform_vars_z_hat.c
 * @brief Implementation of the z^hat transformation of decision values for the BBOB Schwefel problem.
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_vars_z_hat.
 */
typedef struct {
  double *xopt;
  double *z;
  coco_problem_free_function_t old_free_problem;
} transform_vars_z_hat_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_z_hat_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_z_hat_data_t *data;
  coco_problem_t *inner_problem;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_z_hat_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  data->z[0] = x[0];

  for (i = 1; i < problem->number_of_variables; ++i) {
    data->z[i] = x[i] + 0.25 * (x[i - 1] - 2.0 * fabs(data->xopt[i - 1]));
  }
  coco_evaluate_function(inner_problem, data->z, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_z_hat_free(void *thing) {
  transform_vars_z_hat_data_t *data = (transform_vars_z_hat_data_t *) thing;
  coco_free_memory(data->xopt);
  coco_free_memory(data->z);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_z_hat(coco_problem_t *inner_problem, const double *xopt) {
  transform_vars_z_hat_data_t *data;
  coco_problem_t *problem;
  data = (transform_vars_z_hat_data_t *) coco_allocate_memory(sizeof(*data));
  data->xopt = coco_duplicate_vector(xopt, inner_problem->number_of_variables);
  data->z = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_z_hat_free, "transform_vars_z_hat");
  problem->evaluate_function = transform_vars_z_hat_evaluate;
  /* TODO: implement best_parameter transformation if needed in the case of not zero:
     see also issue #814.
  The correct update of best_parameter seems not too difficult and should not anymore
  break the current implementation of the Schwefel function. 
   coco_warning("transform_vars_z_hat(): 'best_parameter' not updated"); 
 
  This:
  
  size_t i;
  if (problem->best_parameter != NULL)
	for (i = 1; i < problem->number_of_variables; ++i)
	  problem->best_parameter[i] -= 0.25 * (problem->best_parameter[i - 1] - 2.0 * fabs(data->xopt[i - 1]));

  should do, but gives
  COCO INFO: ..., d=2, running: f18.Assertion failed: (about_equal_value(hypervolume, 8.1699208579037619e-05)), function test_coco_archive_extreme_solutions, file ./test_coco_archive.c, line 123.

  */
  if (strstr(coco_problem_get_id(inner_problem), "schwefel") == NULL) {
    coco_warning("transform_vars_z_hat(): 'best_parameter' not updated, set to NAN.");
    coco_vector_set_to_nan(problem->best_parameter, problem->number_of_variables);
  }

  return problem;
}
