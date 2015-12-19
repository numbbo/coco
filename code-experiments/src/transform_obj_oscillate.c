#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"

static void transform_obj_oscillate_evaluate(coco_problem_t *self, const double *x, double *y) {
  static const double factor = 0.1;
  size_t i;
  coco_evaluate_function(coco_transformed_get_inner_problem(self), x, y);
  for (i = 0; i < self->number_of_objectives; i++) {
      if (y[i] != 0) {
          double log_y;
          log_y = log(fabs(y[i])) / factor;
          if (y[i] > 0) {
              y[i] = pow(exp(log_y + 0.49 * (sin(log_y) + sin(0.79 * log_y))), factor);
          } else {
              y[i] = -pow(exp(log_y + 0.49 * (sin(0.55 * log_y) + sin(0.31 * log_y))), factor);
          }
      }
  }
}

/**
 * Oscillate the objective value of the inner problem.
 *
 * Caveat: this can change best_parameter and best_value. 
 */
static coco_problem_t *f_transform_obj_oscillate(coco_problem_t *inner_problem) {
  coco_problem_t *self;
  self = coco_transformed_allocate(inner_problem, NULL, NULL);
  self->evaluate_function = transform_obj_oscillate_evaluate;
  /* Compute best value */
  /* Maybe not the most efficient solution */
  transform_obj_oscillate_evaluate(self, self->best_parameter, self->best_value);
  return self;
}
