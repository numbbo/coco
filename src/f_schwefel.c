#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "coco_generics.c"
#include "bbob2009_legacy_code.c"

int instance;

static void f_schwefel_evaluate(coco_problem_t *self, double *x, double *y) {
	size_t i;
	double tmp, condition = 10., penalty = 0., xopt[40], *tmx, *tmpvect, fopt, fadd;
	tmx = coco_allocate_vector(self->number_of_variables);
	tmpvect = coco_allocate_vector(self->number_of_variables);
	assert(self->number_of_objectives == 1);
	y[0] = 0.0;

	/* Initialization */
	fopt = bbob2009_compute_fopt(20, instance);
	int rseed = 20 + 10000 * instance;
	bbob2009_unif(tmpvect, self->number_of_variables, rseed);
	for (i = 0; i < self->number_of_variables; i++)
	{
		xopt[i] = 0.5 * 4.2096874633;
		if (tmpvect[i] - 0.5 < 0)
			xopt[i] *= -1.;
	}
	fadd = fopt;

	/* Transformation in search space */
	for (i = 0; i < self->number_of_variables; ++i)
	{
		tmpvect[i] = 2. * x[i];
		if (xopt[i] < 0.)
			tmpvect[i] *= -1.;
	}

	tmx[0] = tmpvect[0];
	for (i = 1; i < self->number_of_variables; ++i)
	{
		tmx[i] = tmpvect[i] + 0.25 * (tmpvect[i-1] - 2. * fabs(xopt[i-1]));
	}

	for (i = 0; i < self->number_of_variables; ++i)
	{
		tmx[i] -= 2 * fabs(xopt[i]);
		tmx[i] *= pow(sqrt(condition), ((double)i)/((double)(self->number_of_variables - 1)));
		tmx[i] = 100. * (tmx[i] + 2 * fabs(xopt[i]));
	}

	/* Boundary handling*/
	for (i = 0; i < self->number_of_variables; ++i)
	{
		tmp = fabs(tmx[i]) - 500.;
		if (tmp > 0.)
		{
			penalty += tmp * tmp;
		}
	}
	fadd += 0.01 * penalty;

	/* Computation core */
	for (i = 0; i < self->number_of_variables; ++i)
	{
		y[0] += tmx[i] * sin(sqrt(fabs(tmx[i])));
	}
	y[0] = 0.01 * ((418.9828872724339) - y[0] / (double)self->number_of_variables);
	y[0] += fadd;

	/* Free allocated memory */
	coco_free_memory(tmx);
	coco_free_memory(tmpvect);

}

static coco_problem_t *schwefel_problem(const size_t number_of_variables, const int instance_id) {
    size_t i, problem_id_length;

    instance = instance_id;
    /*fprintf(stdout, "%2i\n",
    		instance);
    fflush(stdout);*/
    coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
    problem->problem_name = coco_strdup("schwefel function");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0,
                                 "%s_%02i", "schwefel", (int)number_of_variables);
    problem->problem_id = (char *)coco_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1,
             "%s_%02d", "schwefel", (int)number_of_variables);

    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = f_schwefel_evaluate;

    for (i = 0; i < number_of_variables; ++i) {
        problem->smallest_values_of_interest[i] = -5.0;
        problem->largest_values_of_interest[i] = 5.0;
        problem->best_parameter[i] = 0;
    }
    /* Calculate best parameter value */
    f_schwefel_evaluate(problem, problem->best_parameter, problem->best_value);
    return problem;
}
