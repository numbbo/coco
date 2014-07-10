#include <assert.h>

#include "numbbo.h"
#include "numbbo_internal.h"

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
        if (self->smallest_values_of_interest != NULL)
            numbbo_free_memory(self->smallest_values_of_interest);
        if (self->largest_values_of_interest != NULL)
            numbbo_free_memory(self->largest_values_of_interest);       
        if (self->best_parameter != NULL)
            numbbo_free_memory(self->best_parameter);
        if (self->best_value != NULL)
            numbbo_free_memory(self->best_value);
        if (self->problem_name != NULL)
            numbbo_free_memory(self->problem_name);
        if (self->problem_id != NULL)
            numbbo_free_memory(self->problem_id);
        self->smallest_values_of_interest = NULL;
        self->largest_values_of_interest = NULL;
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

const size_t numbbo_get_number_of_variables(const numbbo_problem_t *self) {
    assert(self != NULL);
    assert(self->problem_id != NULL);
    return self->number_of_variables;
}

const double * numbbo_get_smallest_values_of_interest(const numbbo_problem_t *self) {
    assert(self != NULL);
    assert(self->problem_id != NULL);
    return self->smallest_values_of_interest;
}

const double * numbbo_get_largest_values_of_interest(const numbbo_problem_t *self) {
    assert(self != NULL);
    assert(self->problem_id != NULL);
    return self->largest_values_of_interest;
}

void numbbo_get_initial_solution(const numbbo_problem_t *self, 
                                 double *initial_solution) {
    assert(self != NULL);
    if(self->initial_solution != NULL) {
        self->initial_solution(self, initial_solution);
    } else {
        assert(self->smallest_values_of_interest != NULL);
        assert(self->largest_values_of_interest != NULL);
        for (size_t i = 0; i < self->number_of_variables; ++i)
            initial_solution[i] = 0.5 * (self->smallest_values_of_interest[i] + self->largest_values_of_interest[i]);
    }
}
