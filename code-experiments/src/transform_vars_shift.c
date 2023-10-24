/**
 * @file transform_vars_shift.c
 * @brief Implementation of shifting all decision values by an offset.
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_vars_shift.
 */
typedef struct {
  double *offset;
  double *shifted_x;
  coco_problem_free_function_t old_free_problem;
} transform_vars_shift_data_t;

/**
 * @brief Evaluates the transformed objective function.
 */
static void transform_vars_shift_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  double *cons_values;
  int is_feasible;
  transform_vars_shift_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_shift_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    data->shifted_x[i] = x[i] - data->offset[i];
  }
  
  coco_evaluate_function(inner_problem, data->shifted_x, y);
  
  if (problem->number_of_constraints > 0) {
    cons_values = coco_allocate_vector(problem->number_of_constraints);
    is_feasible = coco_is_feasible(problem, x, cons_values);
    coco_free_memory(cons_values);    
    if (is_feasible)
      assert(y[0] + 1e-13 >= problem->best_value[0]);
  }
  else assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Evaluates the transformed constraint function.
 */
static void transform_vars_shift_evaluate_constraint(coco_problem_t *problem, const double *x, double *y, int update_counter) {
  size_t i;
  transform_vars_shift_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_shift_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    data->shifted_x[i] = x[i] - data->offset[i];
  }
  inner_problem->evaluate_constraint(inner_problem, data->shifted_x, y, update_counter);
}

/**
 * @brief Evaluates the gradient of the transformed function at x
 */
static void transform_vars_shift_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_shift_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_shift_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);
		  
  for (i = 0; i < problem->number_of_variables; ++i) {
    data->shifted_x[i] = x[i] - data->offset[i];
  }
  bbob_evaluate_gradient(inner_problem, data->shifted_x, y);

}

/**
 * @brief Frees the data object.
 */
static void transform_vars_shift_free(void *thing) {
  transform_vars_shift_data_t *data = (transform_vars_shift_data_t *) thing;
  coco_free_memory(data->shifted_x);
  coco_free_memory(data->offset);
}

/**
 * @brief Creates the transformation.
 * 
 * CAVEAT: when shifting the constraint only, the best_value of best_parameter
 *         will get in an inconsistent state.
 */
static coco_problem_t *transform_vars_shift(coco_problem_t *inner_problem,
                                            const double *offset,
                                            const int shift_constraint_only) {
  transform_vars_shift_data_t *data;
  coco_problem_t *problem;
  size_t i;

  data = (transform_vars_shift_data_t *) coco_allocate_memory(sizeof(*data));
  data->offset = coco_duplicate_vector(offset, inner_problem->number_of_variables);
  data->shifted_x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, 
    transform_vars_shift_free, "transform_vars_shift");
    
  if (inner_problem->number_of_objectives > 0 && shift_constraint_only == 0)
    problem->evaluate_function = transform_vars_shift_evaluate_function;
    
  if (inner_problem->number_of_constraints > 0)
    problem->evaluate_constraint = transform_vars_shift_evaluate_constraint;
    
  problem->evaluate_gradient = transform_vars_shift_evaluate_gradient;
  
  /* Update the best parameter */
  for (i = 0; i < problem->number_of_variables; i++)
    problem->best_parameter[i] += data->offset[i];
    
  /* Update the initial solution if any */
  if (problem->initial_solution)
    for (i = 0; i < problem->number_of_variables; i++)
      problem->initial_solution[i] += data->offset[i];
      
  return problem;
}
