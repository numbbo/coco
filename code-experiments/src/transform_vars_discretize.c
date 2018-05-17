/**
 * @file transform_vars_discretize.c
 *
 * @brief Implementation of transforming a continuous problem to a mixed-integer problem by making some
 * of its variables discrete. The entire ROI (including the variables that are not discretized) is first
 * mapped to the ROI of the inner problem. Then, some shifting is performed so that the best_parameter
 * lies in one of the feasible (discrete) values. Caution: this won't work if the inner problem is not
 * defined outside of the ROI!
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_vars_discretize.
 */
typedef struct {
  double *offset;
} transform_vars_discretize_data_t;

/**
 * @brief Evaluates the transformed objective function.
 */
static void transform_vars_discretize_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_discretize_data_t *data;
  coco_problem_t *inner_problem;
  double *discretized_x = coco_allocate_vector(problem->number_of_variables);
  double inner_l, inner_u, outer_l, outer_u;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_discretize_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  /* The solution x already has integer values where needed */
  for (i = 0; i < problem->number_of_variables; ++i) {
    inner_l = inner_problem->smallest_values_of_interest[i];
    inner_u = inner_problem->largest_values_of_interest[i];
    outer_l = problem->smallest_values_of_interest[i];
    outer_u = problem->largest_values_of_interest[i];
    discretized_x[i] = inner_l + (inner_u - inner_l) * (x[i] - outer_l) / (outer_u - outer_l) - data->offset[i];
  }
  
  coco_evaluate_function(inner_problem, discretized_x, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
  coco_free_memory(discretized_x);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_discretize_free(void *thing) {
  transform_vars_discretize_data_t *data = (transform_vars_discretize_data_t *) thing;
  coco_free_memory(data->offset);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_discretize(coco_problem_t *inner_problem,
                                                 const double *smallest_values_of_interest,
                                                 const double *largest_values_of_interest,
                                                 const int *are_variables_integer) {
  transform_vars_discretize_data_t *data;
  coco_problem_t *problem = NULL;
  double inner_l, inner_u, outer_l, outer_u, xopt;
  size_t i;

  data = (transform_vars_discretize_data_t *) coco_allocate_memory(sizeof(*data));
  data->offset = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_discretize_free, "transform_vars_discretize");
  if (problem->are_variables_integer == NULL) {
    problem->are_variables_integer = coco_allocate_vector_int(problem->number_of_variables);
  }

  for (i = 0; i < problem->number_of_variables; i++) {
    assert(smallest_values_of_interest[i] < largest_values_of_interest[i]);
    problem->smallest_values_of_interest[i] = smallest_values_of_interest[i];
    problem->largest_values_of_interest[i] = largest_values_of_interest[i];
    assert((are_variables_integer[i] == 0) || (are_variables_integer[i] == 1));
    problem->are_variables_integer[i] = are_variables_integer[i];
    if (are_variables_integer[i] == 0)
      data->offset[i] = 0;
    else {
      /* Compute the offset in four steps: */
      inner_l = inner_problem->smallest_values_of_interest[i];
      inner_u = inner_problem->largest_values_of_interest[i];
      outer_l = problem->smallest_values_of_interest[i];
      outer_u = problem->largest_values_of_interest[i];
      /* Step 1: Find location of the optimum in the coordinates of the outer problem */
      xopt = outer_l + (outer_u - outer_l) * (inner_problem->best_parameter[i] - inner_l) / (inner_u - inner_l);
      /* Step 2: Round to the closest integer */
      xopt = coco_double_round(xopt);
      problem->best_parameter[i] = xopt;
      /* Step 3: Find the corresponding discretized value in the inner problem */
      xopt = inner_l + (inner_u - inner_l) * (xopt - outer_l) / (outer_u - outer_l);
      /* Step 4: Compute the difference */
      data->offset[i] = xopt - inner_problem->best_parameter[i];
    }
  }
    
  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_vars_discretize_evaluate_function;

  if (problem->number_of_constraints > 0)
    coco_error("transform_vars_discretize(): Constraints not supported yet.");

  problem->evaluate_constraint = NULL; /* TODO? */
  problem->evaluate_gradient = NULL;   /* TODO? */
      
  return problem;
}
