#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "numbbo.h"

#include "numbbo_problem.c"

static void f_rastrigin_evaluate(numbbo_problem_t *self, double *x, double *y) {
    size_t i;
    double sum1 = 0.0, sum2 = 0.0;
    assert(self->number_of_objectives == 1);
    for (i = 0; i < self->number_of_parameters; ++i) {
        sum1 += cos(2 * numbbo_pi * x[i]);
        sum2 += x[i] * x[i];
    }
    y[0] = 10.0 * (self->number_of_parameters - sum1) + sum2;
}

static numbbo_problem_t *rastrigin_problem(const size_t number_of_parameters) {
    size_t i, problem_id_length;
    numbbo_problem_t *problem = numbbo_allocate_problem(number_of_parameters, 1, 0);
    problem->problem_name = numbbo_strdup("rastrigin function");
    problem_id_length = snprintf(NULL, 0, 
                                 "%s_%02i", "rastrigin", (int)number_of_parameters);
    problem->problem_id = (char *)numbbo_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1, 
             "%s_%02i", "rastrigin", (int)number_of_parameters);
    problem->number_of_parameters = number_of_parameters;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = f_rastrigin_evaluate;
    for (i = 0; i < number_of_parameters; ++i) {
        problem->lower_bounds[i] = -5.0;
        problem->upper_bounds[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    /* Calculate best parameter value */
    f_rastrigin_evaluate(problem, problem->best_parameter, problem->best_value);
    
    return problem;
}
