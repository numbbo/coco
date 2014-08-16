#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "coco.h"

#include "coco_problem.c"

/* Number of summands in the Weierstrass problem. */
#define WEIERSTRASS_SUMMANDS 12
typedef struct {
    double f0;
    double ak[WEIERSTRASS_SUMMANDS];
    double bk[WEIERSTRASS_SUMMANDS];
} _wss_problem_t;

static void _weierstrass_evaluate(coco_problem_t *self, double *x, double *y) {
    size_t i, j;
    _wss_problem_t *data = self->data;
    assert(self->number_of_objectives == 1);

    y[0] = 0.0;
    for (i = 0; i < self->number_of_variables; ++i) {
        for (j = 0; j < WEIERSTRASS_SUMMANDS; ++j) {
            y[0] += cos(2 * coco_pi * (x[i] + 0.5) * data->bk[j]) * data->ak[j];
        }
    }
    y[0] = 10.0 * pow(y[0] / self->number_of_variables - data->f0, 3.0);
}

static coco_problem_t *weierstrass_problem(const size_t number_of_variables) {
    size_t i, problem_id_length;
    coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
    _wss_problem_t *data;
    data = coco_allocate_memory(sizeof(*data));

    data->f0 = 0.0;
    for (i = 0; i < WEIERSTRASS_SUMMANDS; ++i) {
        data->ak[i] = pow(0.5, (double)i);
        data->bk[i] = pow(3., (double)i);
        data->f0 += data->ak[i] * cos(2 * coco_pi * data->bk[i] * 0.5);
   }
    
    problem->problem_name = coco_strdup("weierstrass function");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0,
                                 "%s_%02i", "weierstrass",
                                 (int)number_of_variables);
    problem->problem_id = coco_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1,
             "%s_%02d", "weierstrass", (int)number_of_variables);

    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = _weierstrass_evaluate;
    problem->data = data;
    for (i = 0; i < number_of_variables; ++i) {
        problem->smallest_values_of_interest[i] = -5.0;
        problem->largest_values_of_interest[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    
    /* Calculate best parameter value */
    _weierstrass_evaluate(problem, problem->best_parameter, problem->best_value);
    return problem;
}

#undef WEIERSTRASS_SUMMANDS
