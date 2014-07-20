#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"

static void _different_powers_evaluate(coco_problem_t *self, double *x, double *y) {
    size_t i;
    double sum = 0.0;

    assert(self->number_of_objectives == 1);
    for (i = 0; i < self->number_of_variables; ++i) {
        double exponent = 2.0 + (4.0 * i) / (self->number_of_variables - 1.0);
        sum += pow(fabs(x[i]), exponent);
    }
    y[0] = sqrt(sum);
}

static coco_problem_t *different_powers_problem(const size_t number_of_variables) {
    size_t i, problem_id_length;
    coco_problem_t *problem = coco_allocate_problem(number_of_variables,
                                                        1, 0);
    problem->problem_name = coco_strdup("different powers function");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0, 
                                 "%s_%02i", "different powers",
                                 (int)number_of_variables);
    problem->problem_id = coco_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1, 
             "%s_%02d", "different powers", (int)number_of_variables);

    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = _different_powers_evaluate;
    for (i = 0; i < number_of_variables; ++i) {
        problem->smallest_values_of_interest[i] = -5.0;
        problem->largest_values_of_interest[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    /* Calculate best parameter value */
    _different_powers_evaluate(problem, problem->best_parameter, problem->best_value);
    return problem;
}
