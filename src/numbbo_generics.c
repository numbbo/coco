#include <assert.h>

#include "numbbo.h"

void numbbo_evaluate_function(numbbo_problem_t *self, double *x, double *y) {
    /* implements a safer version of self->evaluate(self, x, y) */
    assert(self != NULL);
    assert(self->evaluate_function != NULL);
    self->evaluate_function(self, x, y);
}

void numbbo_evaluate_constraint(numbbo_problem_t *self, double *x, double *y) {
    /* implements a safer version of self->evaluate(self, x, y) */
    assert(self != NULL);
    assert(self->evaluate_constraint != NULL);
    self->evaluate_constraint(self, x, y);
}

void numbbo_recommend_solutions(numbbo_problem_t *self, 
                                double *x, size_t number_of_solutions) {
    assert(self != NULL);
    assert(self->recommend_solutions != NULL);
    self->recommend_solutions(self, x, number_of_solutions);
}

void numbbo_free_problem(numbbo_problem_t *self) {
    assert(self != NULL);
    if (self->free_problem != NULL) {
        self->free_problem(self);
    } else {
        /* Best guess at freeing all relevant structures */
        if (self->lower_bounds != NULL)
            numbbo_free_memory(self->lower_bounds);
        if (self->upper_bounds != NULL)
            numbbo_free_memory(self->upper_bounds);       
        if (self->best_parameter != NULL)
            numbbo_free_memory(self->best_parameter);
        if (self->best_value != NULL)
            numbbo_free_memory(self->best_value);
        if (self->problem_name != NULL)
            numbbo_free_memory(self->problem_name);
        if (self->problem_id != NULL)
            numbbo_free_memory(self->problem_id);
        self->lower_bounds = NULL;
        self->upper_bounds = NULL;
        self->best_parameter = NULL;
        self->best_value = NULL;
        numbbo_free_memory(self);
    }
}

const char *numbbo_get_name(numbbo_problem_t *self) {
    assert(self != NULL);
    assert(self->problem_name != NULL);
    return self->problem_name;
}

const char *numbbo_get_id(numbbo_problem_t *self) {
    assert(self != NULL);
    assert(self->problem_id != NULL);
    return self->problem_id;
}

const int numbbo_get_number_of_parameters(numbbo_problem_t *self) {
    assert(self != NULL);
    assert(self->problem_id != NULL);
    return self->number_of_parameters;
}
