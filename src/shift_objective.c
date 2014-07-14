#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
    double offset;
} shift_objective_state_t;

static void _so_evaluate_function(coco_problem_t *self, double *x, double *y) {
    coco_transformed_problem_t *problem;
    shift_objective_state_t *state; 

    assert(self != NULL);
    problem = (coco_transformed_problem_t *)self;
    assert(problem->inner_problem != NULL);

    assert(problem->state != NULL);
    state = (shift_objective_state_t *)problem->state;

    coco_evaluate_function(problem->inner_problem, x, y);
    y[0] += state->offset;
}

/* Shift the returned objective value of ${inner_problem} by ${amount}. 
 */
coco_problem_t *shift_objective(coco_problem_t *inner_problem,
                                  const double offset) {
    coco_transformed_problem_t *obj =
        coco_allocate_transformed_problem(inner_problem);
    coco_problem_t *problem = (coco_problem_t *)obj;
    shift_objective_state_t *state = 
        (shift_objective_state_t *)coco_allocate_memory(sizeof(*state));
    state->offset = offset;
    obj->state = state;
    problem->evaluate_function = _so_evaluate_function;
    return problem;
}
