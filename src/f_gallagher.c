#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "bbob2009_legacy_code.c"

double *peaks;
int compare_doubles(const void *, const void *);

typedef struct {
	size_t rseed;
	int number_of_peaks;
	double *tmx;
    double *xopt, fopt;
    double **rotation, **Xlocal, **arrScales;
    double fitvalues[2];
    double *arrCondition;
    double *peakvalues;
    size_t *rperm;
} _bbob_gallagher_t;

static void _bbob_gallagher_101_evaluate(coco_problem_t *self, double *x, double *y) {
	size_t i, j, k; /*Loop over dim*/
	double tmp, tmp2;
	double fac = -0.5 / (double)self->number_of_variables;
    double penalty = 0.0;
    _bbob_gallagher_t *data;

    assert(self->number_of_variables > 1);
    assert(self->number_of_objectives == 1);

    data = self->data;
    for (i = 0; i < self->number_of_variables; ++i) {
        double tmp;
        tmp = fabs(x[i]) - 5.0;
        if (tmp > 0.0)
            penalty += tmp * tmp;
    }

    /* Transformation in search space */
    for (i = 0; i < self->number_of_variables; i++) {
    	data->tmx[i] = 0.;
    	for (j = 0; j < self->number_of_variables; j++) {
    		data->tmx[i] += data->rotation[i][j] * x[j];
    	}
    }

    /* Computation core */
    for (i = 0; i < data->number_of_peaks; i++) {
    	tmp2 = 0.;
    	for (j = 0; j < self->number_of_variables; j++) {
    		tmp = (data->tmx[j] - data->Xlocal[j][i]);
    		tmp2 += data->arrScales[i][j] * tmp * tmp;
    	}
    	tmp2 = data->peakvalues[i] * exp(fac * tmp2);
    	y[0] = fmax(y[0], tmp2);
    }
    y[0] = 10. - y[0];
}

static void _bbob_gallagher_free(coco_problem_t *self) {
    _bbob_gallagher_t *data;
    data = self->data;
    coco_free_memory(data->tmx);
    coco_free_memory(data->xopt);
    coco_free_memory(data->rperm);
    coco_free_memory(data->arrCondition);
    coco_free_memory(data->peakvalues);
    bbob2009_free_matrix(data->rotation, self->number_of_variables);
    bbob2009_free_matrix(data->Xlocal, self->number_of_variables);
    bbob2009_free_matrix(data->arrScales, data->number_of_peaks);
    /* Let the generic free problem code deal with all of the
     * coco_problem_t fields.
     */
    self->free_problem = NULL;
    coco_free_problem(self);
}

static coco_problem_t *bbob_gallagher_problem(const size_t number_of_variables,
                                                   const int instance_id, const int number_of_peaks) {
    size_t i, j, k, problem_id_length, rseed;
    coco_problem_t *problem;
    _bbob_gallagher_t *data;
    double maxcondition = 1000.;

    if (number_of_peaks == 101) {
    	rseed = 21 + 10000 * instance_id;
    } else {
    	rseed = 22 + 10000 * instance_id;
    }


    data = coco_allocate_memory(sizeof(*data));
    /* Allocate temporary storage and space for the rotation matrices */
    data->rseed = rseed;
    data->number_of_peaks = number_of_peaks;
    data->tmx = coco_allocate_vector(number_of_variables);
    data->xopt = coco_allocate_vector(number_of_variables);
    data->fitvalues[0] = 1.1;
    data->fitvalues[1] = 9.1;
    data->rotation = bbob2009_allocate_matrix(number_of_variables, number_of_variables);
    data->Xlocal = bbob2009_allocate_matrix(number_of_variables, number_of_peaks);
    data->arrScales = bbob2009_allocate_matrix(number_of_peaks, number_of_variables);
    data->fopt = bbob2009_compute_fopt(21, instance_id);
    bbob2009_compute_xopt(data->xopt, rseed, number_of_variables);
    bbob2009_compute_rotation(data->rotation, rseed + 1000000, number_of_variables);
    problem = coco_allocate_problem(number_of_variables, 1, 0);
    /* Construct a meaningful problem id */
    if (number_of_peaks == 101) {
    	problem->problem_name = coco_strdup("BBOB f21");
    	problem_id_length = snprintf(NULL, 0, "%s_%02i", "bbob2009_f21", (int)number_of_variables);
    	problem->problem_id = coco_allocate_memory(problem_id_length + 1);
    	snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "bbob2009_f21", (int)number_of_variables);
    } else {
    	problem->problem_name = coco_strdup("BBOB f22");
    	problem_id_length = snprintf(NULL, 0, "%s_%02i", "bbob2009_f22", (int)number_of_variables);
    	problem->problem_id = coco_allocate_memory(problem_id_length + 1);
    	snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "bbob2009_f22", (int)number_of_variables);
    }
    
    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->data = data;
    problem->evaluate_function = _bbob_gallagher_101_evaluate;
    problem->free_problem = _bbob_gallagher_free;
    for (i = 0; i < number_of_variables; ++i) {
        problem->smallest_values_of_interest[i] = -5.0;
        problem->largest_values_of_interest[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    
    /* Initialize all the data of the inner problem */
    peaks = coco_allocate_vector(number_of_variables * number_of_peaks);/*Allocated the largest needed size*/
    bbob2009_unif(peaks, number_of_peaks - 1, data->rseed);
    data->rperm = (size_t *)malloc(number_of_peaks-1);/* TODO: consider using another function, must free all these */
    for (i = 0; i < number_of_peaks - 1; i++)
    	data->rperm[i] = i;
    qsort(data->rperm, number_of_peaks - 1, sizeof(size_t), compare_doubles);
    
    /* Random permutation */
    data->arrCondition = coco_allocate_vector(number_of_peaks);
    data->arrCondition[0] = sqrt(maxcondition);
    data->peakvalues = coco_allocate_vector(number_of_peaks);
    data->peakvalues[0] = 10;
    for (i = 1; i < number_of_peaks; i++) {
    	data->arrCondition[i] = pow(maxcondition, (double)(data->rperm[i-1])/((double)(number_of_peaks - 2)));
    	data->peakvalues[i] = (double)(i-1)/(double)(number_of_peaks - 2) * (data->fitvalues[1] - data->fitvalues[0]) + data->fitvalues[0];
    }
    for (i = 0; i < number_of_peaks; i++) {
    	/*bbob2009_unif(peaks, number_of_variables, data->rseed + 1000 * i);*/
    	for (j = 0; j < number_of_variables; j++)
    		data->rperm[j] = j;
    	qsort(data->rperm, number_of_variables, sizeof(int), compare_doubles);
    	for (j = 0; j < number_of_variables; j++) {
    		data->arrScales[i][j] = pow(data->arrCondition[i], ((double)data->rperm[j])/((double)(number_of_variables - 1)) - 0.5);
    	}
    }
    
    bbob2009_unif(peaks, number_of_variables * number_of_peaks, data->rseed);
    for (i = 0; i < number_of_variables; i++) {
    	data->xopt[i] = 0.8 * (10. * peaks[i] -5.);
    	for (j = 0; j < number_of_peaks; j++) {
    		data->Xlocal[i][j] = 0.;
    		for (k = 0; k < number_of_variables; k++) {
    			data->Xlocal[i][j] += data->rotation[i][k] * (10. * peaks[j * number_of_variables + k] - 5.);
    		}
    		if (j == 0){
    			data->Xlocal[i][j] *= 0.8;
    		}
    	}
    }

    /* Calculate best parameter value */
    problem->evaluate_function(problem, problem->best_parameter, problem->best_value);
    free(peaks);/*TODO: right place for doing this?*/
    return problem;
}

int compare_doubles(const void *a, const void *b) {
    double temp = peaks[*(const int*)a] - peaks[*(const int*)b];
    if (temp > 0)
        return 1;
    else if (temp < 0)
        return -1;
    else
        return 0;
}

