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
#include "bbob2009_legacy_code.c"

int rseed;

static void f_stepEllipsoidal_evaluate(coco_problem_t *self, double *x, double *y) {
    size_t i, j;
    static const double condition = 1.0e2;
    static double alpha = 10.;
    double M[40*40], b[40], xopt[40], fopt;
    double **rot;
    double *tmpvect;
    double *z;

    assert(self->number_of_objectives == 1);
    tmpvect = coco_allocate_memory(sizeof(double *));
    z = coco_allocate_memory(sizeof(double *));
    for (i = 0; i < self->number_of_variables; ++i) {
    	if (fabs(x[i]) > 0.5)
    		tmpvect[i] = round(x[i]);
    	else
    		tmpvect[i] = round(0.5 + alpha * tmpvect[i])/alpha;
    }
    rot = bbob2009_allocate_matrix(self->number_of_variables, self->number_of_variables);
    bbob2009_compute_rotation(rot, rseed + 1000000, self->number_of_variables);
    bbob2009_copy_rotation_matrix(rot, M, b, self->number_of_variables);
    bbob2009_free_matrix(rot, self->number_of_variables);
    for (i = 0; i < self->number_of_variables; ++i) {
    	const double *current_row = M + i * self->number_of_variables;
        z[i] = b[i];
        for (j = 0; j < self->number_of_variables; ++j) {
        z[i] += tmpvect[j] * current_row[j];
        }
    }
    /* Computation core */
    y[0] = 0.0;
    for (i = 0; i < self->number_of_variables; ++i) {
        y[0] += pow(condition,
                    ((double)(i - 1))/((double)(self->number_of_variables - 1))) * z[i] * z[i];
    }
    y[0] = 0.1 * fmax(fabs(x[1]) * 1.0e-4, y[0]);
    coco_free_memory(tmpvect);
    coco_free_memory(z);
}

static coco_problem_t *stepEllipsoidal_problem(const size_t number_of_variables, int seed) {
    size_t i, problem_id_length;
    rseed = seed;
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

