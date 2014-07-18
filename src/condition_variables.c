/*
 * Implementation of the BBOB Gamma transformation for variables.
 */
#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
    double *x;
    double alpha;
} _cv_data_t;

static void _cv_evaluate_function(coco_problem_t *self, double *x, double *y) {
    size_t i;
    double exponent;
    _cv_data_t *data;
    coco_problem_t *inner_problem;

    data = coco_get_transform_data(self);
    inner_problem = coco_get_transform_inner_problem(self);
    
    for (i = 0; i < self->number_of_variables; ++i) {
        /* OME: We could precalculate the scaling coefficients if we
         * really wanted to. 
         */
        data->x[i] = pow(data->alpha, 0.5 * i / (self->number_of_variables - 1.0)) * x[i];
    }
    coco_evaluate_function(inner_problem, data->x, y);
}

static void _cv_free_data(void *thing) {
    _cv_data_t *data = thing;
    coco_free_memory(data->x);
}

/**
 * Perform monotone oscillation transformation on input variables.
 */
coco_problem_t *condition_variables(coco_problem_t *inner_problem,
                                    const double alpha) {
    _cv_data_t *data;
    coco_problem_t *self;
    data = coco_allocate_memory(sizeof(*data));
    data->x = coco_allocate_vector(inner_problem->number_of_variables);
    data->alpha = alpha;
    self = coco_allocate_transformed_problem(inner_problem, data, _cv_free_data);
    self->evaluate_function = _cv_evaluate_function;
    return self;
}
