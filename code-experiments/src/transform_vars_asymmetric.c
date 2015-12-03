/*
 * Implementation of the BBOB T_asy transformation for variables.
 */
#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
  double *x;
  double beta;
} transform_vars_asymmetric_data_t;

static void transform_vars_asymmetric_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  double exponent;
  transform_vars_asymmetric_data_t *data;
  coco_problem_t *inner_problem;

  data = coco_transformed_get_data(self);
  inner_problem = coco_transformed_get_inner_problem(self);

  for (i = 0; i < self->number_of_variables; ++i) {
    if (x[i] > 0.0) {
      exponent = 1.0
          + (data->beta * (double) (long) i) / ((double) (long) self->number_of_variables - 1.0) * sqrt(x[i]);
      data->x[i] = pow(x[i], exponent);
    } else {
      data->x[i] = x[i];
    }
  }
  coco_evaluate_function(inner_problem, data->x, y);
}

static void transform_vars_asymmetric_free(void *thing) {
  transform_vars_asymmetric_data_t *data = thing;
  coco_free_memory(data->x);
}

/**
 * Perform monotone oscillation transformation on input variables.
 */
static coco_problem_t *f_transform_vars_asymmetric(coco_problem_t *inner_problem, const double beta) {
  transform_vars_asymmetric_data_t *data;
  coco_problem_t *self;
  data = coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->beta = beta;
  self = coco_transformed_allocate(inner_problem, data, transform_vars_asymmetric_free);
  self->evaluate_function = transform_vars_asymmetric_evaluate;
  return self;
}
