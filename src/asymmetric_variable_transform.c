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
} _avt_data_t;

static void _avt_evaluate_function(coco_problem_t *self, double *x, double *y) {
    size_t i;
    double exponent;
    _avt_data_t *data;
    coco_problem_t *inner_problem;

    data = coco_get_transform_data(self);
    inner_problem = coco_get_transform_inner_problem(self);
    
    for (i = 0; i < self->number_of_variables; ++i) {
        if (x[i] > 0.0) {
            exponent = 1.0 + (data->beta * i) / 
                (self->number_of_variables - 1.0) * sqrt(x[i]);
            data->x[i] = pow(x[i], exponent);
        } else {
            data->x[i] = x[i];
        }
    }
    coco_evaluate_function(inner_problem, data->x, y);
}

static void _avt_free_data(void *thing) {
    _avt_data_t *data = thing;
    coco_free_memory(data->x);
}

/**
 * Perform monotone oscillation transformation on input variables.
 */
coco_problem_t *asymmetric_variable_transform(coco_problem_t *inner_problem,
                                              const double beta) {
    _avt_data_t *data;
    coco_problem_t *self;
    data = coco_allocate_memory(sizeof(*data));
    data->x = coco_allocate_vector(inner_problem->number_of_variables);
    data->beta = beta;
    self = coco_allocate_transformed_problem(inner_problem, data, _avt_free_data);
    self->evaluate_function = _avt_evaluate_function;
    return self;
}
