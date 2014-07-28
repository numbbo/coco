#include <assert.h>

#include "coco.h"
#include "coco_internal.h"

void coco_evaluate_function(coco_problem_t *self, double *x, double *y) {
    /* implements a safer version of self->evaluate(self, x, y) */
    assert(self != NULL);
    assert(self->evaluate_function != NULL);
    self->evaluate_function(self, x, y);
}

void coco_evaluate_constraint(coco_problem_t *self, double *x, double *y) {
    /* implements a safer version of self->evaluate(self, x, y) */
    assert(self != NULL);
    assert(self->evaluate_constraint != NULL);
    self->evaluate_constraint(self, x, y);
}

void coco_recommend_solutions(coco_problem_t *self,
                                double *x, size_t number_of_solutions) {
    assert(self != NULL);
    assert(self->recommend_solutions != NULL);
    self->recommend_solutions(self, x, number_of_solutions);
}

void coco_free_problem(coco_problem_t *self) {
    assert(self != NULL);
    if (self->free_problem != NULL) {
        self->free_problem(self);
    } else {
        /* Best guess at freeing all relevant structures */
        if (self->smallest_values_of_interest != NULL)
            coco_free_memory(self->smallest_values_of_interest);
        if (self->largest_values_of_interest != NULL)
            coco_free_memory(self->largest_values_of_interest);
        if (self->best_parameter != NULL)
            coco_free_memory(self->best_parameter);
        if (self->best_value != NULL)
            coco_free_memory(self->best_value);
        if (self->problem_name != NULL)
            coco_free_memory(self->problem_name);
        if (self->problem_id != NULL)
            coco_free_memory(self->problem_id);
        if (self->data != NULL)
            coco_free_memory(self->data);
        self->smallest_values_of_interest = NULL;
        self->largest_values_of_interest = NULL;
        self->best_parameter = NULL;
        self->best_value = NULL;
        self->data = NULL;
        coco_free_memory(self);
    }
}

const char *coco_get_problem_name(coco_problem_t *self) {
    assert(self != NULL);
    assert(self->problem_name != NULL);
    return self->problem_name;
}

const char *coco_get_problem_id(coco_problem_t *self) {
    assert(self != NULL);
    assert(self->problem_id != NULL);
    return self->problem_id;
}

size_t coco_get_number_of_variables(const coco_problem_t *self) {
    assert(self != NULL);
    assert(self->problem_id != NULL);
    return self->number_of_variables;
}

size_t coco_get_number_of_objectives(const coco_problem_t *self) {
    assert(self != NULL);
    assert(self->problem_id != NULL);
    return self->number_of_objectives;
}

const double * coco_get_smallest_values_of_interest(const coco_problem_t *self) {
    assert(self != NULL);
    assert(self->problem_id != NULL);
    return self->smallest_values_of_interest;
}

const double * coco_get_largest_values_of_interest(const coco_problem_t *self) {
    assert(self != NULL);
    assert(self->problem_id != NULL);
    return self->largest_values_of_interest;
}

void coco_get_initial_solution(const coco_problem_t *self,
                                 double *initial_solution) {
    assert(self != NULL);
    if(self->initial_solution != NULL) {
        self->initial_solution(self, initial_solution);
    } else {
        size_t i;
        assert(self->smallest_values_of_interest != NULL);
        assert(self->largest_values_of_interest != NULL);
        for (i = 0; i < self->number_of_variables; ++i)
            initial_solution[i] = 0.5 * (self->smallest_values_of_interest[i] + self->largest_values_of_interest[i]);
    }
}
