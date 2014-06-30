#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "numbbo.h"

#include "numbbo_problem.c"

static void f_linearSlope_evaluate(numbbo_problem_t *self, double *x, double *y) {
    /*  TODO: considers optimum in [-5]*dim, update for using any Xopt
        TODO: boundary handling
        TODO: shift x using Xopt 
        TODO: replace "y[0] += 0" with "y[0] += fopt" of the instance
     */
    size_t i;
    double *tmpx = (double *)malloc(self->number_of_parameters * sizeof(double));
    static double alpha = 100.;/* */
    assert(self->number_of_objectives == 1 );
    y[0] = 0.0;
    /* move "too" good coordinates back into domain
     original code:
     for (i = 0; i < DIM; i++) {
     if ((Xopt[i] == 5.) && (x[i] > 5))
     tmx[i] = 5.;
     else if ((Xopt[i] == -5.) && (x[i] < -5))
     tmx[i] = -5.;
     else
     tmx[i] = x[i];
     }
     */

    for (i = 0; i < self->number_of_parameters; ++i) {
        //printf("%f",x[i]);
        if (x[i] < -5)
            tmpx[i] = 0;// put fopt at 0
        else
            tmpx[i] = x[i] +5;
    }
    /* COMPUTATION core*/
    for ( i = 0; i < self->number_of_parameters; ++i ){
        /*if (Xopt[i] > 0) {
            y[0] -= pow(sqrt(alpha), ((double)i)/((double)(self->number_of_parameters-1))) * tmpx[i];
        } else {*/
            y[0] += pow(sqrt( alpha ), ( ( double ) i )/( ( double )( self->number_of_parameters - 1))) * tmpx[i];
        /*}*/
    }

    y[0] += 0 ;
    
}

static numbbo_problem_t *linearSlope_problem(const size_t number_of_parameters) {
    size_t i, problem_id_length;
    numbbo_problem_t *problem = numbbo_allocate_problem(number_of_parameters, 1, 0);
    problem->problem_name = numbbo_strdup("linear slope function");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0,
                                 "%s_%02i", "linearSlope", (int)number_of_parameters);
    problem->problem_id = numbbo_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1,
             "%s_%02d", "linearSlope", (int)number_of_parameters);
    
    problem->number_of_parameters = number_of_parameters;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = f_linearSlope_evaluate;
    for (i = 0; i < number_of_parameters; ++i) {
        problem->lower_bounds[i] = -5.0;/*TODO: boundary handling*/
        problem->upper_bounds[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    /* Calculate best parameter value */
    f_linearSlope_evaluate(problem, problem->best_parameter, problem->best_value);
    return problem;
}

