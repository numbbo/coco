/**
 * @file transform_vars_brs.c
 * @brief Implementation of the ominous 's_i scaling' of the BBOB Bueche-Rastrigin problem.
 */

#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_vars_brs.
 */
typedef struct {
  double *x;
} transform_vars_brs_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_brs_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  double factor;
  transform_vars_brs_data_t *data;
  coco_problem_t *inner_problem;

  data = (transform_vars_brs_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    /* Function documentation says we should compute 10^(0.5 *
     * (i-1)/(D-1)). Instead we compute the equivalent
     * sqrt(10)^((i-1)/(D-1)) just like the legacy code.
     */
    factor = pow(sqrt(10.0), (double) (long) i / ((double) (long) problem->number_of_variables - 1.0));
    /* Documentation specifies odd indices and starts indexing
     * from 1, we use all even indices since C starts indexing
     * with 0.
     */
    if (x[i] > 0.0 && i % 2 == 0) {
      factor *= 10.0;
    }
    data->x[i] = factor * x[i];
  }
  coco_evaluate_function(inner_problem, data->x, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_brs_free(void *thing) {
  transform_vars_brs_data_t *data = (transform_vars_brs_data_t *) thing;
  coco_free_memory(data->x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_brs(coco_problem_t *inner_problem) {
  transform_vars_brs_data_t *data;
  coco_problem_t *problem;
  size_t i = 0, zero = 1;
  data = (transform_vars_brs_data_t *) coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_brs_free, "transform_vars_brs");
  problem->evaluate_function = transform_vars_brs_evaluate;
  while (i < inner_problem->number_of_variables && zero) {
      zero = (inner_problem->best_parameter[i] == 0);
      i++;
  }
  if (!zero) {
      coco_warning("transform_vars_brs(): 'best_parameter' not updated");
  }
  return problem;
}
