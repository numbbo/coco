#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "numbbo.h"

#include "numbbo_problem.c"

static void f_differentPowers_evaluate(numbbo_problem_t *self, double *x, double *y) {
    size_t i;
    assert(self->number_of_objectives == 1);
    y[0] = 0.0;
    for (i = 1; i < self->number_of_variables; ++i) {
        y[0] += pow(fabs(x[i]),
                    2 + 4 * (((double)(i - 1))/(self->number_of_variables - 1)));
    }
    y[0] = sqrt(y[0]);
}

static numbbo_problem_t *differentPowers_problem(const size_t number_of_variables) {
    size_t i, problem_id_length;
    numbbo_problem_t *problem = numbbo_allocate_problem(number_of_variables,
                                                        1, 0);
    problem->problem_name = numbbo_strdup("different powers function");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0, 
                                 "%s_%02i", "different powers",
                                 (int)number_of_variables);
    problem->problem_id = numbbo_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1, 
             "%s_%02d", "different powers", (int)number_of_variables);

    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = f_differentPowers_evaluate;
    for (i = 0; i < number_of_variables; ++i) {
        problem->lower_bounds[i] = -5.0;
        problem->upper_bounds[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    /* Calculate best parameter value */
    f_differentPowers_evaluate(problem, problem->best_parameter, problem->best_value);
    return problem;
}
