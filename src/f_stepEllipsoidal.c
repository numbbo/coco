/*
 * f_stepEllipsoidal.c
 *
 *  Created on: Jun 30, 2014
 *      Author: asma
 */
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "coco.h"

#include "coco_problem.c"

static void f_stepEllipsoidal_evaluate(coco_problem_t *self, double *x, double *y) {
    size_t i;
    assert(self->number_of_objectives == 1);

    /* Computing f_pen */
    double f_pen = 0, diff = 0, f_opt = 0;
    static const double condition = 1.0e2;
    for (i = 0; i < self->number_of_variables; ++i){
    	diff = fabs(x[i]) - 5;
    	if (diff > 0){
    		f_pen += diff * diff;
    	}
    }
    /* Computation core */
    y[0] = 0.0;
    for (i = 0; i < self->number_of_variables; ++i) {
        y[0] += pow(condition,
                    ((double)(i - 1))/((double)(self->number_of_variables - 1))) * x[i] * x[i];
    }
    y[0] = 0.1 * fmax(fabs(x[1]) * 1.0e-4, y[0]);
    y[0] += f_pen + f_opt;
}

static coco_problem_t *stepEllipsoidal_problem(const size_t number_of_variables) {
    size_t i, problem_id_length;
    coco_problem_t *problem = coco_allocate_problem(number_of_variables,
                                                        1, 0);
    problem->problem_name = coco_strdup("step ellipsoidal function");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0,
                                 "%s_%02i", "step ellipsoidal",
                                 (int)number_of_variables);
    problem->problem_id = coco_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1,
             "%s_%02d", "step ellipsoidal", (int)number_of_variables);

    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = f_stepEllipsoidal_evaluate;
    for (i = 0; i < number_of_variables; ++i) {
        problem->smallest_values_of_interest[i] = -5.0;
        problem->largest_values_of_interest[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    /* Calculate best parameter value */
    f_stepEllipsoidal_evaluate(problem, problem->best_parameter, problem->best_value);
    return problem;
}

