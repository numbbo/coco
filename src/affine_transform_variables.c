#include <stdbool.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/*
 * Perform an affine transformation of the variable vector:
 *
 *   x |-> Mx + b
 *
 * The matrix M is stored in row-major format.
 */

typedef struct {
    double *M, *b, *x;
} _atv_data_t;

static void _atv_evaluate_function(coco_problem_t *self, double *x, double *y) {
    size_t i, j;
    _atv_data_t *data;
    coco_problem_t *inner_problem;

    data = coco_get_transform_data(self);
    inner_problem = coco_get_transform_inner_problem(self);

    for (i = 0; i < inner_problem->number_of_variables; ++i) {
        /* data->M has self->number_of_variables columns and
         * problem->inner_problem->number_of_variables rows.
         */
        const double *current_row = data->M + i * self->number_of_variables;
        data->x[i] = data->b[i];
        for (j = 0; j < self->number_of_variables; ++j) {
            data->x[i] += x[j] * current_row[j];
        }
    }
    coco_evaluate_function(inner_problem, data->x, y);
}

static void _atv_free_data(void *thing) {
    _atv_data_t *data = thing;
    coco_free_memory(data->M);
    coco_free_memory(data->b);
    coco_free_memory(data->x);
}

/*
 * FIXMEs:
 * - Calculate new smallest/largest values of interest?
 * - Resize bounds vectors if input and output dimensions do not match
 * - problem_id and problem_name need to be adjusted
 */
coco_problem_t *affine_transform_variables(coco_problem_t *inner_problem,
                                           const double *M, const double *b,
                                           const size_t number_of_variables) {
    coco_problem_t *self;
    _atv_data_t *data;
    size_t entries_in_M;

    entries_in_M = inner_problem->number_of_variables * number_of_variables;
    data = coco_allocate_memory(sizeof(*data));
    data->M = coco_duplicate_vector(M, entries_in_M);
    data->b = coco_duplicate_vector(b, inner_problem->number_of_variables);
    data->x = coco_allocate_vector(inner_problem->number_of_variables);

    self = coco_allocate_transformed_problem(inner_problem, data, _atv_free_data);
    self->evaluate_function = _atv_evaluate_function;
    return self;
}
