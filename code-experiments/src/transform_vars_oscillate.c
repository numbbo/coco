/**
 * @file transform_vars_oscillate.c
 * @brief Implementation of oscillating the decision values.
 */

#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_vars_oscillate.
 */
typedef struct {
  double *oscillated_x;
} transform_vars_oscillate_data_t;


/**
 * @brief Multivariate, coordinate-wise, oscillating non-linear transformation.
 */
static transform_vars_oscillate_data_t *tosz(transform_vars_oscillate_data_t *data,
                                              const double *x,
                                              int number_of_variables) {
  size_t i;
  static const double alpha = 0.1;
  double tmp, base, *oscillated_x;

  oscillated_x = data->oscillated_x; /* short cut to make code more readable */
  for (i = 0; i < number_of_variables; ++i) {
    if (x[i] > 0.0) {
      tmp = log(x[i]) / alpha;
      base = exp(tmp + 0.49 * (sin(tmp) + sin(0.79 * tmp)));
      oscillated_x[i] = pow(base, alpha);
    } else if (x[i] < 0.0) {
      tmp = log(-x[i]) / alpha;
      base = exp(tmp + 0.49 * (sin(0.55 * tmp) + sin(0.31 * tmp)));
      oscillated_x[i] = -pow(base, alpha);
    } else {
      oscillated_x[i] = 0.0;
    }
  }
  return data;
}


/**
 * @brief Evaluates the transformed objective functions.
 */
static void transform_vars_oscillate_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  double *cons_values;
  int is_feasible;
  transform_vars_oscillate_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_oscillate_data_t *) coco_problem_transformed_get_data(problem);

  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  data = tosz(data, x, problem->number_of_variables);

  coco_evaluate_function(inner_problem, data->oscillated_x, y);
  
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
 * @brief Evaluates the transformed constraints.
 */
static void transform_vars_oscillate_evaluate_constraint(coco_problem_t *problem, const double *x, double *y) {
  transform_vars_oscillate_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_constraints(problem));
  	return;
  }

  data = (transform_vars_oscillate_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  data = tosz(data, x, problem->number_of_variables);

  coco_evaluate_constraint(inner_problem, data->oscillated_x, y);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_oscillate_free(void *thing) {
  transform_vars_oscillate_data_t *data = (transform_vars_oscillate_data_t *) thing;
  coco_free_memory(data->oscillated_x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_oscillate(coco_problem_t *inner_problem) {
  transform_vars_oscillate_data_t *data;
  coco_problem_t *problem;
  data = (transform_vars_oscillate_data_t *) coco_allocate_memory(sizeof(*data));
  data->oscillated_x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, 
    transform_vars_oscillate_free, "transform_vars_oscillate");
    
  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_vars_oscillate_evaluate_function;
    
  if (inner_problem->number_of_constraints > 0)
    problem->evaluate_constraint = transform_vars_oscillate_evaluate_constraint;

  return problem;
}
