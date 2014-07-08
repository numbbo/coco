#include <assert.h>
#include <numbbo.h>

#include "numbbo_problem.c"

typedef struct {
    double amount;
} shift_objective_state_t;

static void _so_evaluate_function(numbbo_problem_t *self, double *x, double *y) {
    assert(problem->inner_problem != NULL);
    assert(problem->state != NULL);
    numbbo_transformed_problem_t *problem = (numbbo_transformed_problem_t *)self;
    shift_objective_state_t *state = (shift_objective_state_t *)self->state;
    numbbo_evaluate_function(problem->inner_problem, x, y);
    y[0] += state->amount;
}

/* Shift the returned objective value of ${inner_problem} by ${amount}. 
 */
numbbo_problem_t *shift_objective(const numbbo_problem_t *inner_problem,
                                      const double amount) {
    numbbo_transformed_problem_t *problem =
        numbbo_allocate_transformed_problem(inner_problem);
    shift_objective_state_t *state = numbbo_allocate_memory(sizeof(*state));
    state->amount = amount;
    problem->state = state;
    problem->evaluate_function = _so_evaluate_function;
    return problem;
}
