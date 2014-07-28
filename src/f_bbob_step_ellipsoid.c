/*
 * f_bbob_step_ellipsoid.c
 *
 * The BBOB step ellipsoid function intertwins the variable and
 * objective transformations in such a way that it is hard to devise a
 * composition of generic transformations to implement it. In the end
 * one would have to implement several custom transformations which
 * would be used soley by this problem. Therefore we opt to implement
 * it as a monolithic function instead.
 *
 * TODO: It would be nice to have a generic step ellipsoid function to
 * complement this one.
 */
#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "bbob2009_legacy_code.c"

typedef struct {
    double *x, *xx;
    double *xopt, fopt;
    double **rot1, **rot2;
} _bbob_step_ellipsoid_t;

static void _bbob_step_ellipsoid_evaluate(coco_problem_t *self, double *x, double *y) {
    static const double condition = 100;
    static const double alpha = 10.0;
    size_t i, j;
    double penalty = 0.0, x1;
    _bbob_step_ellipsoid_t *data;

    assert(self->number_of_objectives == 1);
    data = self->data;
    for (i = 0; i < self->number_of_variables; ++i) {
        double tmp;
        tmp = fabs(x[i]) - 5.0;
        if (tmp > 0.0)
            penalty += tmp * tmp;
    }

    for (i = 0; i < self->number_of_variables; ++i) {
        double c1;
        data->x[i] = 0.0;
        c1 = sqrt(pow(condition/10., i / (self->number_of_variables - 1.0)));
        for (j = 0; j < self->number_of_variables; ++j) {
            data->x[i] += c1 * data->rot2[i][j] * (x[j] - data->xopt[j]);
        }
    }
    x1 = data->x[0];

    for (i = 0; i < self->number_of_variables; ++i) {
        if (fabs(data->x[i]) > 0.5)
            data->x[i] = round(data->x[i]);
        else
            data->x[i] = round(alpha * data->x[i]) / alpha;
    }

    for (i = 0; i < self->number_of_variables; ++i) {
        data->xx[i] = 0.0;
        for (j = 0; j < self->number_of_variables; ++j) {
            data->xx[i] += data->rot1[i][j] * data->x[j];
        }
    }

    /* Computation core */
    y[0] = 0.0;
    for (i = 0; i < self->number_of_variables; ++i) {
        double exponent;
        exponent = i/(self->number_of_variables - 1.0);
        y[0] += pow(condition, exponent) * data->xx[i] * data->xx[i];;
    }
    y[0] = 0.1 * fmax(fabs(x1) * 1.0e-4, y[0]) + penalty + data->fopt;
}

static void _bbob_step_ellipsoid_free(coco_problem_t *self) {
    _bbob_step_ellipsoid_t *data;
    data = self->data;
    coco_free_memory(data->x);
    coco_free_memory(data->xx);
    bbob2009_free_matrix(data->rot1, self->number_of_variables);
    bbob2009_free_matrix(data->rot2, self->number_of_variables);
    /* Let the generic free problem code deal with all of the
     * coco_problem_t fields.
     */
    self->free_problem = NULL;
    coco_free_problem(self);
}

static coco_problem_t *bbob_step_ellipsoid_problem(const size_t number_of_variables,
                                                   const int instance_id) {
    size_t i, problem_id_length, rseed;
    coco_problem_t *problem;
    _bbob_step_ellipsoid_t *data;

    rseed = 7 + 10000 * instance_id;

    data = coco_allocate_memory(sizeof(*data));
    /* Allocate temporary storage and space for the rotation matrices */
    data->x = coco_allocate_vector(number_of_variables);
    data->xx = coco_allocate_vector(number_of_variables);
    data->xopt = coco_allocate_vector(number_of_variables);
    data->rot1 = bbob2009_allocate_matrix(number_of_variables, number_of_variables);
    data->rot2 = bbob2009_allocate_matrix(number_of_variables, number_of_variables);

    data->fopt = bbob2009_compute_fopt(7, instance_id);
    bbob2009_compute_xopt(data->xopt, rseed, number_of_variables);
    bbob2009_compute_rotation(data->rot1, rseed + 1000000, number_of_variables);
    bbob2009_compute_rotation(data->rot2, rseed, number_of_variables);

    problem = coco_allocate_problem(number_of_variables, 1, 0);
    problem->problem_name = coco_strdup("BBOB f7");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0,
                                 "%s_%02i", "bbob2009_f7",
                                 (int)number_of_variables);
    problem->problem_id = coco_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1,
             "%s_%02d", "bbob2009_f7", (int)number_of_variables);

    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->data = data;
    problem->evaluate_function = _bbob_step_ellipsoid_evaluate;
    problem->free_problem = _bbob_step_ellipsoid_free;
    for (i = 0; i < number_of_variables; ++i) {
        problem->smallest_values_of_interest[i] = -5.0;
        problem->largest_values_of_interest[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    /* Calculate best parameter value */
    problem->evaluate_function(problem, problem->best_parameter, problem->best_value);
    return problem;
}
