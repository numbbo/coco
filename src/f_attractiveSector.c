/*
 * f_attractiveSector.c
 *
 *  Created on: Jun 30, 2014
 *      Author: asma
 */
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "numbbo.h"

#include "numbbo_problem.c"

static void f_attractiveSector_evaluate(numbbo_problem_t *self, double *x, double *y) {
    size_t i;
    double condition;
    static const double exp = 0.9;
    assert(self->number_of_objectives == 1);
    y[0] = 0.0;
    for (i = 0; i < self->number_of_variables; ++i) {
    	if (self->best_parameter[i] * x[i] > 0){
    		condition = 1.0e2;
    	}
    	else{
    		condition = 1;
    	}
        y[0] += condition * condition * x[i] * x[i];
    }
    y[0] = pow(y[0], exp);
}

static numbbo_problem_t *attractiveSector_problem(const size_t number_of_variables) {
    size_t i, problem_id_length;
    numbbo_problem_t *problem = numbbo_allocate_problem(number_of_variables,
                                                        1, 0);
    problem->problem_name = numbbo_strdup("attractive sector function");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0,
                                 "%s_%02i", "attractive sector",
                                 (int)number_of_variables);
    problem->problem_id = numbbo_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1,
             "%s_%02d", "attractive sector", (int)number_of_variables);

    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = f_attractiveSector_evaluate;
    for (i = 0; i < number_of_variables; ++i) {
        problem->smallest_values_of_interest[i] = -5.0;
        problem->largest_values_of_interest[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    /* Calculate best parameter value */
    f_attractiveSector_evaluate(problem, problem->best_parameter, problem->best_value);
    return problem;
}

