#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "numbbo.h"

#include "numbbo_problem.c"

static void f_linearSlope_evaluate(numbbo_problem_t *self, double *x, double *y) {
	const static double alpha = 100.0;
	double fadd = 0;
	double *tmpx = (double*) numbbo_allocate_vector(self->number_of_parameters);
	size_t i;
	assert(self->number_of_objectives == 1);
	y[0] = 0.0;
	// all for loops can be merged into one for efficient code, but we lose on lisibility
	for (i = 0; i < self->number_of_parameters; ++i) { //puts para
		fadd += pow(sqrt(alpha),
				((double) i) / ((double) (self->number_of_parameters - 1)));
		if (self->best_parameter[i] > 0.)
			self->best_parameter[i] = 5.;
		else if (self->best_parameter[i] < 0.)
			self->best_parameter[i] = -5.;
	}
	fadd *= 5.;
    
    /* move "too" good coordinates back into domain*/
	for (i = 0; i < self->number_of_parameters; ++i) {
		if ((self->best_parameter[i] == 5.) && (x[i] > 5))
			tmpx[i] = 5.;
		else if ((self->best_parameter[i] == -5.) && (x[i] < -5))
			tmpx[i] = -5.;
		else
			tmpx[i] = x[i];
	}
    
	/* COMPUTATION core*/
	for (i = 0; i < self->number_of_parameters; ++i) {
		if (self->best_parameter[i] > 0) {
			y[0] -= pow(sqrt(alpha),
					((double) i) / ((double) (self->number_of_parameters - 1)))
					* tmpx[i];
		} else {
			y[0] += pow(sqrt(alpha),
					((double) i) / ((double) (self->number_of_parameters - 1)))
					* tmpx[i];
		}

	}
    y[0]+=fadd;
	numbbo_free_memory(tmpx);
}

static numbbo_problem_t *linearSlope_problem(const size_t number_of_parameters) {
	size_t i, problem_id_length;
	numbbo_problem_t *problem = numbbo_allocate_problem(number_of_parameters, 1,
			0);
	problem->problem_name = numbbo_strdup("linear slope function");
	/* Construct a meaningful problem id */
	problem_id_length = snprintf(NULL, 0, "%s_%02i", "linearSlope",
			(int )number_of_parameters);
	problem->problem_id = (char *) numbbo_allocate_memory(
			problem_id_length + 1);
	snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d",
			"linearSlope", (int )number_of_parameters);

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
	f_linearSlope_evaluate(problem, problem->best_parameter,
			problem->best_value);
	return problem;
}

