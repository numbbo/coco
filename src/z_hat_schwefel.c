#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
    double *xopt;
    double *z;
    coco_free_function_t old_free_problem;
} _z_hat_data_t;

static void _z_hat_evaluate_function(coco_problem_t *self, double *x, double *y) {
    size_t i;
    _z_hat_data_t *data;
    coco_problem_t *inner_problem;

    data = coco_get_transform_data(self);
    inner_problem = coco_get_transform_inner_problem(self);

    data->z[0] = x[0];

    for (i = 1; i < self->number_of_variables; ++i) {
        data->z[i] = x[i] + 0.25 * (x[i-1] - 2 * fabs(data->xopt[i-1]));
    }
    coco_evaluate_function(inner_problem, data->z, y);
}

static void _z_hat_free_data(void *thing) {
    _z_hat_data_t *data = thing;
    coco_free_memory(data->xopt);
    coco_free_memory(data->z);
}

/* Compute the vector {z^hat} for f_schwefel
 */
coco_problem_t *z_hat(coco_problem_t *inner_problem,
                                const double *xopt) {
    _z_hat_data_t *data;
    coco_problem_t *self;

    data = coco_allocate_memory(sizeof(*data));
    data->xopt = coco_duplicate_vector(xopt, inner_problem->number_of_variables);
    data->z = coco_allocate_vector(inner_problem->number_of_variables);

    self = coco_allocate_transformed_problem(inner_problem, data, _z_hat_free_data);
    self->evaluate_function = _z_hat_evaluate_function;
    return self;
}



