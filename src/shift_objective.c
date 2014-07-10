#include <assert.h>

#include "numbbo.h"
#include "numbbo_problem.c"

typedef struct {
    double offset;
} shift_objective_state_t;

static void _so_evaluate_function(numbbo_problem_t *self, double *x, double *y) {
    numbbo_transformed_problem_t *problem = (numbbo_transformed_problem_t *)self;
    assert(problem->inner_problem != NULL);
    assert(problem->state != NULL);
    shift_objective_state_t *state = (shift_objective_state_t *)problem->state;

    numbbo_evaluate_function(problem->inner_problem, x, y);
    y[0] += state->offset;
}

/* Shift the returned objective value of ${inner_problem} by ${amount}. 
 */
numbbo_problem_t *shift_objective(numbbo_problem_t *inner_problem,
                                  const double offset) {
    numbbo_transformed_problem_t *obj =
        numbbo_allocate_transformed_problem(inner_problem);
    numbbo_problem_t *problem = (numbbo_problem_t *)obj;
    shift_objective_state_t *state = 
        (shift_objective_state_t *)numbbo_allocate_memory(sizeof(*state));
    state->offset = offset;
    obj->state = state;
    problem->evaluate_function = _so_evaluate_function;
    return problem;
}
