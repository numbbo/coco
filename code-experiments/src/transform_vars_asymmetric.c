/**
 * @file transform_vars_asymmetric.c
 * @brief Implementation of performing an asymmetric transformation on decision values.
 */

#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_vars_asymmetric.
 */
typedef struct {
  double *x;
  double beta;
} transform_vars_asymmetric_data_t;

/**
 * @brief Evaluates the transformed function.
 */
static void transform_vars_asymmetric_evaluate_function(coco_problem_t *problem, 
                                                        const double *x, 
                                                        double *y) {
  size_t i;
  double exponent, *cons_values;
  int is_feasible;
  transform_vars_asymmetric_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_asymmetric_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    if (x[i] > 0.0) {
      exponent = 1.0
          + ((data->beta * (double) (long) i) / ((double) (long) problem->number_of_variables - 1.0)) * sqrt(x[i]);
      data->x[i] = pow(x[i], exponent);
    } else {
      data->x[i] = x[i];
    }
  }
  
  coco_evaluate_function(inner_problem, data->x, y);
  
  if (problem->number_of_constraints > 0) {
    cons_values = coco_allocate_vector(problem->number_of_constraints);
    is_feasible = coco_is_feasible(problem, x, cons_values, 0.0);
    coco_free_memory(cons_values);    
    if (is_feasible)
      assert(y[0] + 1e-13 >= problem->best_value[0]);
  }
  else assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Evaluates the transformed constraint.
 */
static void transform_vars_asymmetric_evaluate_constraint(coco_problem_t *problem, 
                                                          const double *x, 
                                                          double *y) {
  size_t i;
  double exponent;
  transform_vars_asymmetric_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_constraints(problem));
  	return;
  }

  data = (transform_vars_asymmetric_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    if (x[i] > 0.0) {
      exponent = 1.0
          + ((data->beta * (double) (long) i) / ((double) (long) problem->number_of_variables - 1.0)) * sqrt(x[i]);
      data->x[i] = pow(x[i], exponent);
    } else {
      data->x[i] = x[i];
    }
  }
  coco_evaluate_constraint(inner_problem, data->x, y);
}

static void transform_vars_asymmetric_free(void *thing) {
  transform_vars_asymmetric_data_t *data = (transform_vars_asymmetric_data_t *) thing;
  coco_free_memory(data->x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_asymmetric(coco_problem_t *inner_problem, const double beta) {
  
  size_t i;
  int is_feasible;
  double alpha, *cons_values;
  transform_vars_asymmetric_data_t *data;
  coco_problem_t *problem;
  
  data = (transform_vars_asymmetric_data_t *) coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->beta = beta;
  problem = coco_problem_transformed_allocate(inner_problem, data, 
    transform_vars_asymmetric_free, "transform_vars_asymmetric");
    
  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_vars_asymmetric_evaluate_function;
    
  if (inner_problem->number_of_constraints > 0) {
	  
    problem->evaluate_constraint = transform_vars_asymmetric_evaluate_constraint;
    
    /* Check if the initial solution remains feasible after
     * the transformation. If not, do a backtracking
     * towards the origin until it becomes feasible.
     */
    if (inner_problem->initial_solution) {
      cons_values = coco_allocate_vector(problem->number_of_constraints);
      is_feasible = coco_is_feasible(problem, inner_problem->initial_solution, cons_values, 0.0);
      alpha = 0.9;
      i = 0;
      while (!is_feasible) {
        problem->initial_solution[i] *= alpha;
        is_feasible = coco_is_feasible(problem, problem->initial_solution, cons_values, 0.0);
        i = (i + 1) % inner_problem->number_of_variables;
      }
      coco_free_memory(cons_values);
    }
  }
  
  if (inner_problem->number_of_objectives > 0 && coco_problem_best_parameter_not_zero(inner_problem)) {
    coco_warning("transform_vars_asymmetric(): 'best_parameter' not updated, set to NAN");
    coco_vector_set_to_nan(inner_problem->best_parameter, inner_problem->number_of_variables);
  }
  return problem;
}
