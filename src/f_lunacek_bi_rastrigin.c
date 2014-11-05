#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "bbob2009_legacy_code.c"

typedef struct {
    double *x_hat, *z;
    double *xopt, fopt;
    double **rot1, **rot2;
    size_t rseed;
    coco_free_function_t old_free_problem;
} _bbob_lunacek_bi_rastrigin_t;

static void _bbob_lunacek_bi_rastrigin_evaluate(coco_problem_t *self, double *x, double *y) {
	static const double condition = 100.;
	size_t i, j;
	double penalty = 0.0;
	static const double mu0 = 2.5;
    static const double d = 1.;
    double s = 1. - 0.5 / (sqrt((double)(self->number_of_variables + 20)) - 4.1);
    double mu1 = -sqrt((mu0 * mu0 - d) / s);
    _bbob_lunacek_bi_rastrigin_t *data;
    double *tmpvect, sum1 = 0., sum2 = 0., sum3 = 0.;

    assert(self->number_of_variables > 1);
    assert(self->number_of_objectives == 1);
    data = self->data;
    for (i = 0; i < self->number_of_variables; ++i) {
        double tmp;
        tmp = fabs(x[i]) - 5.0;
        if (tmp > 0.0)
            penalty += tmp * tmp;
    }
    /* Computing xopt  */
    tmpvect = coco_allocate_vector(self->number_of_variables);
    bbob2009_gauss(tmpvect, self->number_of_variables, data->rseed);
    for (i = 0; i < self->number_of_variables; ++i) {
    	data->xopt[i] = 0.5 * mu0;
    	if (tmpvect[i] < 0.){
    		data->xopt[i] *= -1.;
    	}
    }
    /* x_hat */
    for (i = 0; i < self->number_of_variables; ++i) {
    	data->x_hat[i] = 2. * x[i];
    	if (data->xopt[i] < 0.){
    		data->x_hat[i] *= -1.;
    	}
    }
    /* affine transformation */
    for (i = 0; i < self->number_of_variables; ++i) {
    	double c1;
        tmpvect[i] = 0.0;
        c1 = pow(sqrt(condition), ((double)i) / (double)(self->number_of_variables - 1));
        for (j = 0; j < self->number_of_variables; ++j) {
        	tmpvect[i] += c1 * data->rot2[i][j] * (data->x_hat[j] - mu0);
        }
    }
    for (i = 0; i < self->number_of_variables; ++i) {
    	data->z[i] = 0;
        for (j = 0; j < self->number_of_variables; ++j) {
        	data->z[i] += data->rot1[i][j] * tmpvect[j];
        }
    }
    /* Computation core */
    for (i = 0; i < self->number_of_variables; ++i) {
    	sum1 += (data->x_hat[i] - mu0) * (data->x_hat[i] - mu0);
    	sum2 += (data->x_hat[i] - mu1) * (data->x_hat[i] - mu1);
    	sum3 += cos(2 * M_PI * data->z[i]);
    }
    y[0] = fmin(sum1, d * (double)self->number_of_variables + s * sum2) + 10. * ((double)self->number_of_variables - sum3) + 1e4 * penalty;
    coco_free_memory(tmpvect);
}

static void _bbob_lunacek_bi_rastrigin_free(coco_problem_t *self) {
    _bbob_lunacek_bi_rastrigin_t *data;
    data = self->data;
    coco_free_memory(data->x_hat);
    coco_free_memory(data->z);
    coco_free_memory(data->xopt);
    bbob2009_free_matrix(data->rot1, self->number_of_variables);
    bbob2009_free_matrix(data->rot2, self->number_of_variables);
    /* Let the generic free problem code deal with all of the
     * coco_problem_t fields.
     */
    self->free_problem = NULL;
    coco_free_problem(self);
}

static coco_problem_t *bbob_lunacek_bi_rastrigin_problem(const size_t number_of_variables,
                                                   const int instance_id) {
    size_t i, problem_id_length, rseed;
    coco_problem_t *problem;
    _bbob_lunacek_bi_rastrigin_t *data;

    rseed = 24 + 10000 * instance_id;

    data = coco_allocate_memory(sizeof(*data));
    /* Allocate temporary storage and space for the rotation matrices */
    data->x_hat = coco_allocate_vector(number_of_variables);
    data->z = coco_allocate_vector(number_of_variables);
    data->xopt = coco_allocate_vector(number_of_variables);
    data->rot1 = bbob2009_allocate_matrix(number_of_variables, number_of_variables);
    data->rot2 = bbob2009_allocate_matrix(number_of_variables, number_of_variables);
    data->rseed = rseed;

    data->fopt = bbob2009_compute_fopt(24, instance_id);
    bbob2009_compute_xopt(data->xopt, rseed, number_of_variables);
    bbob2009_compute_rotation(data->rot1, rseed + 1000000, number_of_variables);
    bbob2009_compute_rotation(data->rot2, rseed, number_of_variables);

    problem = coco_allocate_problem(number_of_variables, 1, 0);
    problem->problem_name = coco_strdup("BBOB f24");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0,
                                 "%s_%02i", "bbob2009_f24",
                                 (int)number_of_variables);
    problem->problem_id = coco_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1,
             "%s_%02d", "bbob2009_f24", (int)number_of_variables);

    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->data = data;
    problem->evaluate_function = _bbob_lunacek_bi_rastrigin_evaluate;
    problem->free_problem = _bbob_lunacek_bi_rastrigin_free;
    for (i = 0; i < number_of_variables; ++i) {
        problem->smallest_values_of_interest[i] = -5.0;
        problem->largest_values_of_interest[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    /* Calculate best parameter value */
    problem->evaluate_function(problem, problem->best_parameter, problem->best_value);
    return problem;
}
