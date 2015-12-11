/*
 * Implementation of the ominous 's_i scaling' of the BBOB Bueche-Rastrigin
 * function.
 */
#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
  double *x;
} transform_vars_brs_data_t;

static void transform_vars_brs_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  double factor;
  transform_vars_brs_data_t *data;
  coco_problem_t *inner_problem;

  data = coco_transformed_get_data(self);
  inner_problem = coco_transformed_get_inner_problem(self);

  for (i = 0; i < self->number_of_variables; ++i) {
    /* Function documentation says we should compute 10^(0.5 *
     * (i-1)/(D-1)). Instead we compute the equivalent
     * sqrt(10)^((i-1)/(D-1)) just like the legacy code.
     */
    factor = pow(sqrt(10.0), (double) (long) i / ((double) (long) self->number_of_variables - 1.0));
    /* Documentation specifies odd indexes and starts indexing
     * from 1, we use all even indexes since C starts indexing
     * with 0.
     */
    if (x[i] > 0.0 && i % 2 == 0) {
      factor *= 10.0;
    }
    data->x[i] = factor * x[i];
  }
  coco_evaluate_function(inner_problem, data->x, y);
}

static void transform_vars_brs_free(void *thing) {
  transform_vars_brs_data_t *data = thing;
  coco_free_memory(data->x);
}

/**
 * Perform monotone oscillation transformation on input variables.
 */
static coco_problem_t *f_transform_vars_brs(coco_problem_t *inner_problem) {
  transform_vars_brs_data_t *data;
  coco_problem_t *self;
  data = coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  self = coco_transformed_allocate(inner_problem, data, transform_vars_brs_free);
  self->evaluate_function = transform_vars_brs_evaluate;
  return self;
}
