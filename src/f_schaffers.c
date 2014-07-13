#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "numbbo.h"

#include "numbbo_problem.c"

/* Schaffers F7 function, transformations not implemented for the moment  */

static void f_schaffers_evaluate(numbbo_problem_t *self, double *x, double *y) {
    size_t i;
    assert(self->number_of_objectives == 1);

    /* Computation core */
    y[0] = 0.0;
    for (i = 0; i < self->number_of_variables - 1; ++i) {
    	const double tmp = x[i] * x[i] + x[i + 1] * x[i + 1];
    	y[0] += pow(tmp, 0.25) * (1 + pow(sin(50. * pow(tmp, 0.1)), 2.));

    }
    y[0] = pow((1./(self->number_of_variables - 1)) * y[0], 2.);
}

static numbbo_problem_t *schaffers_problem(const size_t number_of_variables) {
    size_t i, problem_id_length;
    numbbo_problem_t *problem = numbbo_allocate_problem(number_of_variables,
                                                        1, 0);
    problem->problem_name = numbbo_strdup("schaffers function");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0,
                                 "%s_%02i", "schaffers",
                                 (int)number_of_variables);
    problem->problem_id = numbbo_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1,
             "%s_%02d", "schaffers", (int)number_of_variables);

    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = f_schaffers_evaluate;
    for (i = 0; i < number_of_variables; ++i) {
        problem->smallest_values_of_interest[i] = -5.0;
        problem->largest_values_of_interest[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    /* Calculate best parameter value */
    f_schaffers_evaluate(problem, problem->best_parameter, problem->best_value);
    return problem;
}


