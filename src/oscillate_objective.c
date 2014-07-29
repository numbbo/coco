#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"

static void _oo_evaluate_function(coco_problem_t *self, double *x, double *y) {
    coco_evaluate_function(coco_get_transform_inner_problem(self), x, y);
    if (y[0] > 0){
    	y[0] = pow(exp(log(y[0])/0.1 + 0.49*(sin(log(y[0])/0.1) + sin(0.79*log(y[0])/0.1))), 0.1);
    }
    else if (y[0] < 0){
    	y[0] = -pow(exp(log(-y[0])/0.1 + 0.49*(sin(0.55 * log(-y[0])/0.1) + sin(0.31*log(-y[0])/0.1))), 0.1);
    }
}

/**
 * Oscillate the objective value of the inner problem.
 */
coco_problem_t *oscillate_objective(coco_problem_t *inner_problem) {
    coco_problem_t *self;
    self = coco_allocate_transformed_problem(inner_problem, NULL, NULL);
    self->evaluate_function = _oo_evaluate_function;
    return self;
}
