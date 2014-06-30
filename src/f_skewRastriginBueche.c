#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "numbbo.h"

#include "numbbo_problem.c"

static void f_skewRastriginBueche_evaluate(numbbo_problem_t *self, double *x, double *y) {
    /*
     TODO: boundary handling
     TODO: use Xopt
     TODO: shift x using Xopt
     TODO: replace "y[0] += 0" with "y[0] += fopt" of the instance
     */
    size_t i;
    double tmp = 0., tmp2 = 0.;
    assert( self->number_of_objectives == 1 );
    y[0] = 0.0;
    for (i = 0; i < self->number_of_parameters; ++i ) {
        tmp += cos( 2 * numbbo_pi * x[i] );
        tmp2 += x[i] * x[i];
    }
    y[0] = 10 * (self->number_of_parameters - tmp) + tmp2 + 0;/*TODO: introduce the penalization term f_pen=sum( (max(0,|x|-5.)**2 ) */
    y[0] += 0;/* 0-> fopt*/
    
}

static numbbo_problem_t *skewRastriginBueche_problem(const size_t number_of_parameters) {
    size_t i, problem_id_length;
    numbbo_problem_t *problem = numbbo_allocate_problem(number_of_parameters, 1, 0);
    problem->problem_name = numbbo_strdup("skew Rastrigin-Bueche function");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0,
                                 "%s_%02i", "skewRastriginBueche", (int)number_of_parameters);
    problem->problem_id = numbbo_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1,
             "%s_%02d", "skewRastriginBueche", (int)number_of_parameters);
    
    problem->number_of_parameters = number_of_parameters;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = f_skewRastriginBueche_evaluate;
    for (i = 0; i < number_of_parameters; ++i) {
        problem->lower_bounds[i] = -5.0;
        problem->upper_bounds[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    /* Calculate best parameter value */
    f_skewRastriginBueche_evaluate(problem, problem->best_parameter, problem->best_value);
    return problem;
}

