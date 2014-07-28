#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

static void _linear_slope_evaluate(coco_problem_t *self, double *x, double *y) {
    static const double alpha = 100.0;
    size_t i;
    assert(self->number_of_objectives == 1);
    y[0] = 0.0;
    for (i = 0; i < self->number_of_variables; ++i) {
        double base, exponent, si;

        base = sqrt(alpha);
        exponent = i * 1.0 / (self->number_of_variables - 1);
        if (self->best_parameter[i] > 0.0) {
            si = pow(base, exponent);
        } else {
            si = -pow(base, exponent);
        }
        y[0] += 5.0 * fabs(si) - si * x[i];
    }
}

static coco_problem_t *linear_slope_problem(const size_t number_of_variables,
                                            const double *best_parameter) {
    size_t i, problem_id_length;
    coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
    problem->problem_name = coco_strdup("linear slope function");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0, "%s_%02i", "linearSlope",
                                 (int)number_of_variables);
    problem->problem_id = (char *) coco_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d",
             "linear_slope", (int)number_of_variables);

    problem->evaluate_function = _linear_slope_evaluate;
    for (i = 0; i < number_of_variables; ++i) {
        problem->smallest_values_of_interest[i] = -5.0;
        problem->largest_values_of_interest[i] = 5.0;
        if (best_parameter[i] < 0.0) {
            problem->best_parameter[i] = problem->smallest_values_of_interest[i];
        } else {
            problem->best_parameter[i] = problem->largest_values_of_interest[i];
        }
    }
    /* Calculate best parameter value */
    _linear_slope_evaluate(problem, problem->best_parameter,
                           problem->best_value);
    return problem;
}
