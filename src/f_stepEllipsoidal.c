/*
 * f_stepEllipsoidal.c
 *
 *  Created on: Jun 30, 2014
 *      Author: asma
 */
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "numbbo.h"

#include "numbbo_problem.c"

static void f_stepEllipsoidal_evaluate(numbbo_problem_t *self, double *x, double *y) {
    size_t i;
    assert(self->number_of_objectives == 1);

    /* Computing f_pen */
    double f_pen = 0, diff = 0, f_opt = 0;
    static const double condition = 1.0e2;
    for (i = 0; i < self->number_of_parameters; ++i){
    	diff = fabs(x[i]) - 5;
    	if (diff > 0){
    		f_pen += diff * diff;
    	}
    }
    /* Computation core */
    y[0] = 0.0;
    for (i = 0; i < self->number_of_parameters; ++i) {
        y[0] += pow(condition, ((double)(i - 1))/((double)(self->number_of_parameters - 1))) * x[i] * x[i];
    }
    y[0] = 0.1 * fmax(fabs(x[1]) * 1.0e-4, y[0]);
    y[0] += f_pen + f_opt;
}

static numbbo_problem_t *stepEllipsoidal_problem(const size_t number_of_parameters) {
    size_t i, problem_id_length;
    numbbo_problem_t *problem = numbbo_allocate_problem(number_of_parameters, 1, 0);
    problem->problem_name = numbbo_strdup("step ellipsoidal function");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0,
                                 "%s_%02i", "step ellipsoidal", (int)number_of_parameters);
    problem->problem_id = numbbo_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1,
             "%s_%02d", "step ellipsoidal", (int)number_of_parameters);

    problem->number_of_parameters = number_of_parameters;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = f_stepEllipsoidal_evaluate;
    for (i = 0; i < number_of_parameters; ++i) {
        problem->lower_bounds[i] = -5.0;
        problem->upper_bounds[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    /* Calculate best parameter value */
    f_stepEllipsoidal_evaluate(problem, problem->best_parameter, problem->best_value);
    return problem;
}

