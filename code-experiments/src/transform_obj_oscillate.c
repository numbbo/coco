#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"

static void transform_obj_oscillate_evaluate(coco_problem_t *problem, const double *x, double *y) {
  static const double factor = 0.1;
  size_t i;
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  for (i = 0; i < problem->number_of_objectives; i++) {
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
  assert(y[0] + 1e-13 >= self->best_value[0]);
}

/**
 * Oscillate the objective value of the inner problem.
 *
 * Caveat: this can change best_parameter and best_value. 
 */
static coco_problem_t *f_transform_obj_oscillate(coco_problem_t *inner_problem) {
  coco_problem_t *problem;
  problem = coco_problem_transformed_allocate(inner_problem, NULL, NULL);
  problem->evaluate_function = transform_obj_oscillate_evaluate;
  /* Compute best value */
  /* Maybe not the most efficient solution */
  transform_obj_oscillate_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
