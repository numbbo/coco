/**
 * @file transform_vars_x_hat.c
 * @brief Implementation of multiplying the decision values by the vector 1+-.
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"

/**
 * @brief Data type for transform_vars_x_hat.
 */
typedef struct {
  long seed;
  double *x;
  coco_problem_free_function_t old_free_problem;
} transform_vars_x_hat_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_x_hat_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_x_hat_data_t *data;
  coco_problem_t *inner_problem;
  data = coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);
  do {
    bbob2009_unif(data->x, problem->number_of_variables, data->seed);

    for (i = 0; i < problem->number_of_variables; ++i) {
      if (data->x[i] - 0.5 < 0.0) {
        data->x[i] = -x[i];
      } else {
        data->x[i] = x[i];
      }
    }
    coco_evaluate_function(inner_problem, data->x, y);
    assert(y[0] + 1e-13 >= problem->best_value[0]);
  } while (0);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_x_hat_free(void *thing) {
  transform_vars_x_hat_data_t *data = thing;
  coco_free_memory(data->x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_x_hat(coco_problem_t *inner_problem, long seed) {
  transform_vars_x_hat_data_t *data;
  coco_problem_t *problem;
  size_t i;

  data = (transform_vars_x_hat_data_t *) coco_allocate_memory(sizeof(*data));
  data->seed = seed;
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_x_hat_free);
  problem->evaluate_function = transform_vars_x_hat_evaluate;
  /* Dirty way of setting the best parameter of the transformed f_schwefel... */
  bbob2009_unif(data->x, problem->number_of_variables, data->seed);
  for (i = 0; i < problem->number_of_variables; ++i) {
      if (data->x[i] - 0.5 < 0.0) {
          problem->best_parameter[i] = -0.5 * 4.2096874633;
      } else {
          problem->best_parameter[i] = 0.5 * 4.2096874633;
      }
  }
  return problem;
}
