#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "numbbo.h"

#include "numbbo_problem.c"

static void f_schwefel_evaluate(numbbo_problem_t *self, double *x, double *y) {
    size_t i;
    assert(self->number_of_objectives == 1);
    y[0] = 0.0;
    double *tmp = (double *)numbbo_allocate_memory(sizeof(double));
    double *tmpx = (double *)numbbo_allocate_memory(sizeof(double));

    // Computing x^
    for (i = 0; i < self->number_of_parameters; ++i){
    	tmpx[i] = 2. * x[i];
    	if (self->best_parameter[i] < 0){
    		tmpx[i] *= -1;
    	}
    }

    tmp[0] = tmpx[0]; //z
    for (i = 1; i < self->number_of_parameters; ++i){
    	tmp[i] = tmpx[i] + 0.25 * (tmpx[i - 1] - self->best_parameter[i - 1]); //z
    }

    for (i = 0; i < self->number_of_parameters; ++i){
    	tmp[i] -= self->best_parameter[i]; //z
    }

    /* Transformation missing */

    for (i = 0; i < self->number_of_parameters; ++i){
        	tmp[i] = 100 * (tmp[i] + self->best_parameter[i]);
    }


    /* Computing f_pen */
     double f_pen = 0, diff = 0, f_opt = 0;
     static const double condition = 1.0e2;
     for (i = 0; i < self->number_of_parameters; ++i){
    	 diff = fabs(tmp[i]/100.) - 5;
    	 if (diff > 0){
    		 f_pen += diff * diff;
    	 }
     }

    /* Computation core */
    for (i = 0; i < self->number_of_parameters; ++i){
    	y[0] += tmp[i] * sin(sqrt(fabs(tmp[i])));
    }
    y[0] = -1./((double)self->number_of_parameters) * y[0] + 4.189828872724339 + 100 * f_pen + f_opt;

    numbbo_free_memory(tmp);
    numbbo_free_memory(tmpx);
}

static numbbo_problem_t *schwefel_problem(const size_t number_of_parameters) {
    size_t i, problem_id_length;
    numbbo_problem_t *problem = numbbo_allocate_problem(number_of_parameters, 1, 0);
    problem->problem_name = numbbo_strdup("schwefel function");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0,
                                 "%s_%02i", "schwefel", (int)number_of_parameters);
    problem->problem_id = (char *)numbbo_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1,
             "%s_%02d", "schwefel", (int)number_of_parameters);

    problem->number_of_parameters = number_of_parameters;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = f_schwefel_evaluate;

    uint32_t seed; // must have the same value as the seed above
    numbbo_random_state_t *state = numbbo_new_random(seed);

    for (i = 0; i < number_of_parameters; ++i) {
        problem->lower_bounds[i] = -5.0;
        problem->upper_bounds[i] = 5.0;
        problem->best_parameter[i] = 0.5 * 4.2096874633;
        if (numbbo_uniform_random(state) < 0.5){
        	problem->best_parameter[i] *= -1;
        }
    }
    /* Calculate best parameter value */
    f_schwefel_evaluate(problem, problem->best_parameter, problem->best_value);
    return problem;
}
