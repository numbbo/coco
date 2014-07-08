#include <stdbool.h>
#include <assert.h>

#include "numbbo.h"
#include "numbbo_problem.c"

typedef struct {
    double amount;
    double *tmp;
    numbbo_free_function_t old_free_problem;
} shift_variables_state_t;

static void _sv_evaluate_function(numbbo_problem_t *self, double *x, double *y) {
    numbbo_transformed_problem_t *problem = (numbbo_transformed_problem_t *)self;
    assert(problem->inner_problem != NULL);
    assert(problem->state != NULL);
    shift_variables_state_t *state = (shift_variables_state_t *)problem->state;

    for (int i = 0; i < self->number_of_parameters; ++i) {
        state->tmp[i] = x[i] + state->amount;
    }
    numbbo_evaluate_function(problem->inner_problem, state->tmp, y);
}

static void _sv_free_problem(numbbo_problem_t *self) {
    numbbo_transformed_problem_t *problem = (numbbo_transformed_problem_t *)self;
    assert(problem->inner_problem != NULL);
    assert(problem->state != NULL);
    shift_variables_state_t *state = (shift_variables_state_t *)problem->state;
    
    numbbo_free_memory(state->tmp);
    state->old_free_problem(self);
}

/* Shift all variables of ${inner_problem} by ${amount}.
 */
numbbo_problem_t *shift_variables(numbbo_problem_t *inner_problem,
                                  const double amount,
                                  const bool shift_bounds) {
    numbbo_transformed_problem_t *obj =
        numbbo_allocate_transformed_problem(inner_problem);
    numbbo_problem_t *problem = (numbbo_problem_t *)obj;

    shift_variables_state_t *state = numbbo_allocate_memory(sizeof(*state));
    state->amount = amount;
    state->tmp = (double *)numbbo_allocate_memory(sizeof(double));
    state->old_free_problem = problem->free_problem;
    obj->state = state;
    problem->evaluate_function = _sv_evaluate_function;
    problem->free_problem = _sv_free_problem;
    return problem;
}
