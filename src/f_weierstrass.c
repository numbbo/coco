#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "numbbo.h"

#include "numbbo_problem.c"

static void f_weierstrass_evaluate(numbbo_problem_t *self, double *x, double *y) {
    size_t i, k; /* check what size_t exactly is and whether it can be cast to double */
    assert(self->number_of_objectives == 1);

    /* Computing f_pen */
    double f_pen = 0, diff, f_opt = 0; /* f_opt may need to be changed*/
    static const double condition = 1.0e2;
    for (i = 0; i < self->number_of_variables; ++i){
    	diff = fabs(x[i]) - 5;
    	if (diff > 0){
    		f_pen += diff * diff;
    	}
    }
    /* Computation core */

    /* Computing f_0 */
    double f_0 = 0;
    for (i = 0; i < 12; i++){
        	f_0 += pow(2., ((double)-i)) * cos(2 * M_PI * pow(3., ((double)i)) * 0.5);
        }

    y[0] = 0.0;
    for (i = 0; i < self->number_of_variables; ++i) {
    	for (k = 0; k < 12; k++){
    		y[0] += pow(2., ((double)-k)) * cos(2 * M_PI * pow(3., ((double)k)) * (x[i] + 0.5));
    	}

    }
    y[0] = 10 * pow((1./((double)self->number_of_variables)) * y[0] - f_0, 3.) + 10./((double)self->number_of_variables) * f_pen + f_opt;
}

static numbbo_problem_t *weierstrass_problem(const size_t number_of_variables) {
    size_t i, problem_id_length;
    numbbo_problem_t *problem = numbbo_allocate_problem(number_of_variables,
                                                        1, 0);
    problem->problem_name = numbbo_strdup("weierstrass function");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0,
                                 "%s_%02i", "weierstrass",
                                 (int)number_of_variables);
    problem->problem_id = numbbo_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1,
             "%s_%02d", "weierstrass", (int)number_of_variables);

    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = f_weierstrass_evaluate;
    for (i = 0; i < number_of_variables; ++i) {
        problem->smallest_values_of_interest[i] = -5.0;
        problem->largest_values_of_interest[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    /* Calculate best parameter value */
    f_weierstrass_evaluate(problem, problem->best_parameter, problem->best_value);
    return problem;
}


