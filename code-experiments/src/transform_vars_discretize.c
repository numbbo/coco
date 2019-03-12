/**
 * @file transform_vars_discretize.c
 *
 * @brief Implementation of transforming a continuous problem to a mixed-integer problem by making some
 * of its variables discrete. The integer variables are considered as bounded (any variable outside the
 * decision space is mapped to the closest boundary point), while the continuous ones are treated as
 * unbounded.
 *
 * @note The first problem->number_of_integer_variables are integer, while the rest are continuous.
 *
 * The discretization works as follows. Consider the case where the interval [l, u] of the inner problem
 * needs to be discretized to n integer values of the outer problem. First, [l, u] is discretized to n
 * integers by placing the integers so that there is a (u-l)/(n+1) distance between them (and the border
 * points). Then, the transformation is shifted so that the optimum aligns with the closest integer. In
 * this way, we make sure that if the optimum is within [l, u], so are all the shifted points.
 *
 * When evaluating such a problem, the x values of the integer variables are first discretized. Any value
 * x < 0 is mapped to 0 and any value x > (n-1) is mapped to (n-1).
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
  double *discretized_x;
  double l, u, inner_l, inner_u, outer_l, outer_u;
  int n;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
    coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
    return;
  }

  data = (transform_vars_discretize_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  /* Transform x to fit in the discretized space */
  discretized_x = coco_duplicate_vector(x, problem->number_of_variables);
  for (i = 0; i < problem->number_of_integer_variables; ++i) {
    outer_l = problem->smallest_values_of_interest[i];
    outer_u = problem->largest_values_of_interest[i];
    l = inner_problem->smallest_values_of_interest[i];
    u = inner_problem->largest_values_of_interest[i];
    n = coco_double_to_int(outer_u) - coco_double_to_int(outer_l) + 1; /* number of integer values in this coordinate */
    assert(n > 1);
    inner_l = l + (u - l) / (n + 1);
    inner_u = u - (u - l) / (n + 1);
    /* Make sure you the bounds are respected */
    discretized_x[i] = coco_double_round(x[i]);
    if (discretized_x[i] < outer_l)
      discretized_x[i] = outer_l;
    if (discretized_x[i] > outer_u)
      discretized_x[i] = outer_u;
    discretized_x[i] = inner_l + (inner_u - inner_l) * (discretized_x[i] - outer_l) / (outer_u - outer_l) - data->offset[i];
  }

  coco_evaluate_function(inner_problem, discretized_x, y);
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
                                                 const size_t number_of_integer_variables) {
  transform_vars_discretize_data_t *data;
  coco_problem_t *problem = NULL;
  double l, u, inner_l, inner_u, outer_l, outer_u;
  double outer_xopt, inner_xopt, inner_approx_xopt;
  const double precision_offset = 1e-7; /* Needed to avoid issues with rounding doubles */
  int n;
  size_t i;

  data = (transform_vars_discretize_data_t *) coco_allocate_memory(sizeof(*data));
  data->offset = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_discretize_free, "transform_vars_discretize");
  assert(number_of_integer_variables > 0);
  problem->number_of_integer_variables = number_of_integer_variables;

  for (i = 0; i < problem->number_of_variables; i++) {
    assert(smallest_values_of_interest[i] < largest_values_of_interest[i]);
    problem->smallest_values_of_interest[i] = smallest_values_of_interest[i];
    problem->largest_values_of_interest[i] = largest_values_of_interest[i];
    data->offset[i] = 0;
    if (i < number_of_integer_variables) {
      /* Compute the offset for integer variables */
      outer_l = problem->smallest_values_of_interest[i];
      outer_u = problem->largest_values_of_interest[i];
      l = inner_problem->smallest_values_of_interest[i];
      u = inner_problem->largest_values_of_interest[i];
      n = coco_double_to_int(outer_u) - coco_double_to_int(outer_l) + 1; /* number of integer values */
      assert(n > 1);
      inner_l = l + (u - l) / (n + 1);
      inner_u = u - (u - l) / (n + 1);
      /* Find the location of the optimum in the coordinates of the outer problem */
      inner_xopt = inner_problem->best_parameter[i];
      outer_xopt = outer_l + (outer_u - outer_l) * (inner_xopt - inner_l) / (inner_u - inner_l);
      outer_xopt = coco_double_round(outer_xopt + precision_offset);
      /* Make sure you the bounds are respected */
      if (outer_xopt < outer_l)
        outer_xopt = outer_l;
      if (outer_xopt > outer_u)
        outer_xopt = outer_u;
      problem->best_parameter[i] = outer_xopt;
      /* Find the value corresponding to outer_xopt in the coordinates of the inner problem */
      inner_approx_xopt = inner_l + (inner_u - inner_l) * (outer_xopt - outer_l) / (outer_u - outer_l);
      /* Compute the difference between the inner_approx_xopt and inner_xopt */
      data->offset[i] = inner_approx_xopt - inner_xopt;
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
