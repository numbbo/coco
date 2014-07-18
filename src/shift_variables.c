
#include <stdbool.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
    double *offset;
    double *shifted_x;
    coco_free_function_t old_free_problem;
} _sv_data_t;

static void _sv_evaluate_function(coco_problem_t *self, double *x, double *y) {
    size_t i;
    _sv_data_t *data;
    coco_problem_t *inner_problem;

    data = coco_get_transform_data(self);
    inner_problem = coco_get_transform_inner_problem(self);

    for (i = 0; i < self->number_of_variables; ++i) {
        data->shifted_x[i] = x[i] - data->offset[i];
    }
    coco_evaluate_function(inner_problem, data->shifted_x, y);
}

static void _sv_free_data(void *thing) {
    _sv_data_t *data = thing;
    coco_free_memory(data->shifted_x);
    coco_free_memory(data->offset);
}

/* Shift all variables of ${inner_problem} by ${amount}.
 */
coco_problem_t *shift_variables(coco_problem_t *inner_problem,
                                const double *offset,
                                const bool shift_bounds) {
    _sv_data_t *data;
    coco_problem_t *self;
    if (shift_bounds)
        coco_error("shift_bounds not implemented.");

    data = coco_allocate_memory(sizeof(*data));
    data->offset = coco_duplicate_vector(offset, inner_problem->number_of_variables);
    data->shifted_x = coco_allocate_vector(inner_problem->number_of_variables);

    self = coco_allocate_transformed_problem(inner_problem, data, _sv_free_data);
    self->evaluate_function = _sv_evaluate_function;
    return self;
}
