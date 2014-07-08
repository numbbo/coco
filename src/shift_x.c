#include <assert.h>
#include <numbbo.h>

#include "numbbo_problem.c"

typedef struct {
    double amount;
    double *tmp;
} shift_x_state_t;

static void _sx_evaluate_function(numbbo_problem_t *self, double *x, double *y) {
    assert(problem->inner_problem != NULL);
    assert(problem->state != NULL);
    numbbo_transformed_problem_t *problem = (numbbo_transformed_problem_t *)self;
    shift_x_state_t *state = (shift_x_state_t *)self->state;
    state->tmp = (double *)numbbo_allocate_memory(sizeof(double)); // how to free the memory properly?
    int i;
    for (i = 0; i < self->number_of_parameters; ++i) {
            state->tmp[i] = x[i] + state->amount;
        }
    numbbo_evaluate_function(problem->inner_problem, state->tmp, y);
}

/* Shift the returned objective value of ${inner_problem} by ${amount}.
 */
numbbo_problem_t *shift_x(const numbbo_problem_t *inner_problem,
                                      const double amount) {
    numbbo_transformed_problem_t *problem =
        numbbo_allocate_transformed_problem(inner_problem);
    shift_x_state_t *state = numbbo_allocate_memory(sizeof(*state));
    state->amount = amount;
    state->tmp = NULL;
    problem->state = state;
    problem->evaluate_function = _sx_evaluate_function;
    return problem;
}
