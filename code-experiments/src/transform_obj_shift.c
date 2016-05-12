/**
 * @file transform_obj_shift.c
 * @brief Implementation of shifting the objective value by the given offset.
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_obj_shift.
 */
typedef struct {
  double offset;
} transform_obj_shift_data_t;

/**
 * @brief Evaluates the transformed function.
 */
static void transform_obj_shift_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  transform_obj_shift_data_t *data;
  double *cons_values;
  int is_feasible;
  size_t i;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
   coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
   return;
  }
  
  data = (transform_obj_shift_data_t *) coco_problem_transformed_get_data(problem);
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  
  for (i = 0; i < problem->number_of_objectives; i++) {
    y[i] += data->offset;
  }
  
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
 * @brief Evaluates the transformed constraint
 */
static void transform_obj_shift_evaluate_constraint(coco_problem_t *problem, const double *x, double *y) {
  transform_obj_shift_data_t *data;
  size_t i;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_constraints(problem));
  	return;
  }
  
  data = (transform_obj_shift_data_t *) coco_problem_transformed_get_data(problem);
  coco_evaluate_constraint(coco_problem_transformed_get_inner_problem(problem), x, y);
  
  for (i = 0; i < problem->number_of_constraints; i++) {
      y[i] += data->offset;
  }
}

/**
 * @brief Evaluates the gradient of the transformed function at x
 */
static void transform_obj_shift_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }
  
  coco_evaluate_gradient(coco_problem_transformed_get_inner_problem(problem), x, y);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_obj_shift(coco_problem_t *inner_problem, const double offset) {
  coco_problem_t *problem;
  transform_obj_shift_data_t *data;
  size_t i;
  data = (transform_obj_shift_data_t *) coco_allocate_memory(sizeof(*data));
  data->offset = offset;

  problem = coco_problem_transformed_allocate(inner_problem, data, 
    NULL, "transform_obj_shift");
    
  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_obj_shift_evaluate_function;
    
  if (inner_problem->number_of_constraints > 0)
    problem->evaluate_constraint = transform_obj_shift_evaluate_constraint;
    
  problem->evaluate_gradient = transform_obj_shift_evaluate_gradient;
  
  for (i = 0; i < problem->number_of_objectives; i++)
    problem->best_value[0] += offset;
    
  return problem;
}
