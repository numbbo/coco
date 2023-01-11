/**
 * @file transform_vars_enforce_box.c
 * @brief Enforce box constraint, returning NaN if box is violated.
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Evaluates the transformed objective function.
 */
static void
transform_vars_enforce_box_evaluate_function(coco_problem_t *problem,
                                             const double *x, double *y) {
  size_t i;
  double *cons_values;
  int is_feasible;
  coco_problem_t *inner_problem;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
    coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
    return;
  }

  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    if ((x[i] < problem->smallest_values_of_interest[i]) ||
        (x[i] > problem->largest_values_of_interest[i])) {
      /* Box constraint is violated, return Inf */
      coco_vector_set_to_inf(y, coco_problem_get_number_of_objectives(problem));
      return;
    }
  }

  coco_evaluate_function(inner_problem, x, y);

  if (problem->number_of_constraints > 0) {
    cons_values = coco_allocate_vector(problem->number_of_constraints);
    is_feasible = coco_is_feasible(problem, x, cons_values);
    coco_free_memory(cons_values);
    if (is_feasible) {
      assert(y[0] + 1e-13 >= problem->best_value[0]);
    }
  } else {
    assert(y[0] + 1e-13 >= problem->best_value[0]);
  }
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *
transform_vars_enforce_box(coco_problem_t *inner_problem) {
  coco_problem_t *problem;

  problem = coco_problem_transformed_allocate(inner_problem, NULL, NULL,
                                              "transform_vars_enforce_box");

  problem->evaluate_function = transform_vars_enforce_box_evaluate_function;

  return problem;
}
