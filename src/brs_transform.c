/*
 * Implementation of the ominuous 's_i scaling' of the BBOB Bueche-Rastrigin function.
 */
#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
    double *x;
} _brs_data_t;

static void _brs_evaluate_function(coco_problem_t *self, double *x, double *y) {
    size_t i;
    double factor;
    _brs_data_t *data;
    coco_problem_t *inner_problem;

    data = coco_get_transform_data(self);
    inner_problem = coco_get_transform_inner_problem(self);

    for (i = 0; i < self->number_of_variables; ++i) {
        /* Function documentation says we should compute 10^(0.5 *
         * (i-1)/(D-1)). Instead we compute the equivalent
         * sqrt(10)^((i-1)/(D-1)) just like the legacy code.
         */
        factor = pow(sqrt(10.0),  i / (self->number_of_variables - 1.0));
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

static void _brs_free_data(void *thing) {
    _brs_data_t *data = thing;
    coco_free_memory(data->x);
}

/**
 * Perform monotone oscillation transformation on input variables.
 */
coco_problem_t *brs_transform(coco_problem_t *inner_problem) {
    _brs_data_t *data;
    coco_problem_t *self;
    data = coco_allocate_memory(sizeof(*data));
    data->x = coco_allocate_vector(inner_problem->number_of_variables);
    self = coco_allocate_transformed_problem(inner_problem, data, _brs_free_data);
    self->evaluate_function = _brs_evaluate_function;
    return self;
}
