#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "numbbo.h"

#include "numbbo_problem.c"

static void f_ellipsoid_evaluate(numbbo_problem_t *self, double *x, double *y) {
    size_t i;
    static const double condition = 1.0e6;    
    assert(self->number_of_objectives == 1);
    y[0] = 0.0;
    for (i = 0; i < self->number_of_variables; ++i) {
        const double c1 = (double)(i) / (double)(self->number_of_variables - 1);
        y[0] += pow(condition, c1) * x[i] * x[i];
    }
}

static numbbo_problem_t *ellipsoid_problem(const size_t number_of_variables) {
    size_t i, problem_id_length;
    numbbo_problem_t *problem = numbbo_allocate_problem(number_of_variables,
                                                        1, 0);
    problem->problem_name = numbbo_strdup("ellipsoid function");
    problem_id_length = snprintf(NULL, 0, 
                                 "%s_%02i", "ellipsoid",
                                 (int)number_of_variables);
    problem->problem_id = (char *)numbbo_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1, 
             "%s_%02i", "ellipsoid", (int)number_of_variables);
    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = f_ellipsoid_evaluate;
    for (i = 0; i < number_of_variables; ++i) {
        problem->smallest_values_of_interest[i] = -5.0;
        problem->largest_values_of_interest[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    /* Calculate best parameter value */
    f_ellipsoid_evaluate(problem, problem->best_parameter, problem->best_value);

    return problem;
}
