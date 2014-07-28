#include <assert.h>

#include "coco.h"
#include "coco_problem.c"


static void _puv_evaluate_function(coco_problem_t *self, double *x, double *y) {
    coco_transformed_problem_t *problem = (coco_transformed_problem_t *)self;
    assert(problem->inner_problem != NULL);
    assert(problem->state != NULL);

    coco_evaluate_function(problem->inner_problem, x, y);
    double penalty = 0.0;
    const double *lower_bounds = problem->smallest_values_of_interest;
    const double *upper_bounds = problem->largest_values_of_interest;
    for (size_t i = 0; i < problem->number_of_variables; ++i) {
        assert(lower_bounds[i] < upper_bounds[i]);
        const double c1 = x[i] - upper_bounds[i];
        const double c2 = lower_bounds[i] - x[i];
        if (c1 > 0.0) {
            penalty += c1 * c1;
        } else if (c2 > 0.0) {
            penalty += c2 * c2;
        }
    }
    for (size_t i = 0; i < problem->number_of_objectives; ++i) {
        y[i] += penalty;
    }
}

/**
 * penalize_uninteresting_values(inner_problem):
 *
 * Add a penalty to all evaluations outside of the region of interest
 * of ${inner_problem}.
 */
coco_problem_t *penalize_uninteresting_values(coco_problem_t *inner_problem) {
    assert(inner_problem != NULL);
    assert(offset != NULL);

    size_t number_of_variables = inner_problem->number_of_variables;
    coco_transformed_problem_t *obj =
        coco_allocate_transformed_problem(inner_problem);
    coco_problem_t *problem = (coco_problem_t *)obj;

    problem->evaluate_function = _puv_evaluate_function;
    problem->free_problem = _puv_free_problem;
    return problem;
}
