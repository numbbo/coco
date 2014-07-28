/*
 * Implementation of the BBOB T_osz transformation for variables.
 */

#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
    double *oscillated_x;
} _ov_data_t;

static void _ov_evaluate_function(coco_problem_t *self, double *x, double *y) {
    static const double alpha = 0.1;
    double tmp, base, *oscillated_x;
    size_t i;
    _ov_data_t *data;
    coco_problem_t *inner_problem;

    data = coco_get_transform_data(self);
    oscillated_x = data->oscillated_x; /* short cut to make code more readable */
    inner_problem = coco_get_transform_inner_problem(self);

    for (i = 0; i < self->number_of_variables; ++i) {
        if (x[i] > 0.0) {
            tmp = log(x[i]) / alpha;
            base = exp(tmp + 0.49 * (sin(tmp) + sin(0.79 * tmp)));
            oscillated_x[i] = pow(base, alpha);
        } else if (x[i] < 0.0) {
            tmp = log(-x[i]) / alpha;
            base = exp(tmp + 0.49 * (sin(0.55 * tmp) + sin(0.31 * tmp)));
            oscillated_x[i] = -pow(base, alpha);
        } else {
            oscillated_x[i] = 0.0;
        }
    }
    coco_evaluate_function(inner_problem, oscillated_x, y);
}

static void _ov_free_data(void *thing) {
    _ov_data_t *data = thing;
    coco_free_memory(data->oscillated_x);
}

/**
 * Perform monotone oscillation transformation on input variables.
 */
coco_problem_t *oscillate_variables(coco_problem_t *inner_problem) {
    _ov_data_t *data;
    coco_problem_t *self;
    data = coco_allocate_memory(sizeof(*data));
    data->oscillated_x = coco_allocate_vector(inner_problem->number_of_variables);

    self = coco_allocate_transformed_problem(inner_problem, data, _ov_free_data);
    self->evaluate_function = _ov_evaluate_function;
    return self;
}
