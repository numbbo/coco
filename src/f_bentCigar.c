#include <stdio.h>
#include <assert.h>

#include "coco.h"

#include "coco_problem.c"

static void f_bentCigar_evaluate(coco_problem_t *self, double *x, double *y) {
    size_t i;
    static const double condition = 1.0e6;
    assert(self->number_of_objectives == 1);
    y[0] = 0.0;
    for (i = 1; i < self->number_of_variables; ++i) {
        y[0] += x[i] * x[i];
    }
    y[0] *= condition;
    y[0] += x[0] * x[0];
}

static coco_problem_t *bentCigar_problem(const size_t number_of_variables) {
    size_t i, problem_id_length;
    coco_problem_t *problem = coco_allocate_problem(number_of_variables,
                                                        1, 0);
    problem->problem_name = coco_strdup("bent cigar function");
    /* Construct a meaningful problem id */
    problem_id_length = snprintf(NULL, 0, 
                                 "%s_%02i", "bent cigar",
                                 (int)number_of_variables);
    problem->problem_id = coco_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1, 
             "%s_%02d", "sphere", (int)number_of_variables);

    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = 1;
    problem->number_of_constraints = 0;
    problem->evaluate_function = f_bentCigar_evaluate;
    for (i = 0; i < number_of_variables; ++i) {
        problem->smallest_values_of_interest[i] = -5.0;
        problem->largest_values_of_interest[i] = 5.0;
        problem->best_parameter[i] = 0.0;
    }
    /* Calculate best parameter value */
    f_bentCigar_evaluate(problem, problem->best_parameter, problem->best_value);
    return problem;
}

