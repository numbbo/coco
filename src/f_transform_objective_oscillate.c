#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"

static void private_evaluate_function_too(coco_problem_t *self, const double *x, double *y) {
  static const double factor = 0.1;
  coco_evaluate_function(coco_get_transform_inner_problem(self), x, y);
  if (y[0] != 0) {
    double log_y;
    log_y = log(fabs(y[0])) / factor;
    if (y[0] > 0) {
      y[0] = pow(exp(log_y + 0.49 * (sin(log_y) + sin(0.79 * log_y))), factor);
    } else {
      y[0] = -pow(exp(log_y + 0.49 * (sin(0.55 * log_y) + sin(0.31 * log_y))),
                  factor);
    }
  }
}

/**
 * Oscillate the objective value of the inner problem.
 *
 * Caveat: this can change best_parameter and best_value. 
 */
static coco_problem_t *f_transform_objective_oscillate(coco_problem_t *inner_problem) {
  coco_problem_t *self;
  self = coco_allocate_transformed_problem(inner_problem, NULL, NULL);
  self->evaluate_function = private_evaluate_function_too;
  return self;
}
