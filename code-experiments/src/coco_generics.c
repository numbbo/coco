#include <assert.h>
#include <stddef.h>

#include "coco.h"
#include "coco_internal.h"

void coco_evaluate_function(coco_problem_t *self, const double *x, double *y) {
  /* implements a safer version of self->evaluate(self, x, y) */
  assert(self != NULL);
  assert(self->evaluate_function != NULL);
  self->evaluate_function(self, x, y);
  self->evaluations++; /* each derived class has its own counter, only the most outer will be visible */
#if 1
  /* A little bit of bookkeeping */
  if (y[0] < self->best_observed_fvalue[0]) {
    self->best_observed_fvalue[0] = y[0];
    self->best_observed_evaluation[0] = self->evaluations;
  }
#endif
}

long coco_problem_get_evaluations(coco_problem_t *self) {
  assert(self != NULL);
  return self->evaluations;
}

#if 1  /* tentative */
double coco_problem_get_best_observed_fvalue1(const coco_problem_t *self) {
  assert(self != NULL);
  return self->best_observed_fvalue[0];
}
double coco_problem_get_final_target_fvalue1(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->best_value != NULL);
  assert(self->final_target_delta != NULL);
  return self->best_value[0] + self->final_target_delta[0];
}
#endif

void coco_evaluate_constraint(coco_problem_t *self, const double *x, double *y) {
  /* implements a safer version of self->evaluate(self, x, y) */
  assert(self != NULL);
  assert(self->evaluate_constraint != NULL);
  self->evaluate_constraint(self, x, y);
}

void coco_recommend_solutions(coco_problem_t *self, const double *x, size_t number_of_solutions) {
  assert(self != NULL);
  assert(self->recommend_solutions != NULL);
  self->recommend_solutions(self, x, number_of_solutions);
}

void coco_problem_free(coco_problem_t *self) {
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
    if (self->problem_type != NULL)
      coco_free_memory(self->problem_type);
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

const char *coco_problem_get_name(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_name != NULL);
  return self->problem_name;
}

const char *coco_problem_get_id(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_id != NULL);
  return self->problem_id;
}

const char *coco_problem_get_type(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_type != NULL);
  return self->problem_type;
}

size_t coco_problem_get_dimension(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_id != NULL);
  return self->number_of_variables;
}

size_t coco_problem_get_number_of_objectives(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_id != NULL);
  return self->number_of_objectives;
}

size_t coco_problem_get_number_of_constraints(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_id != NULL);
  return self->number_of_constraints;
}

const double *coco_problem_get_smallest_values_of_interest(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_id != NULL);
  return self->smallest_values_of_interest;
}

const double *coco_problem_get_largest_values_of_interest(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_id != NULL);
  return self->largest_values_of_interest;
}

void coco_problem_get_initial_solution(const coco_problem_t *self, double *initial_solution) {
  assert(self != NULL);
  if (self->initial_solution != NULL) {
    self->initial_solution(self, initial_solution);
  } else {
    size_t i;
    assert(self->smallest_values_of_interest != NULL);
    assert(self->largest_values_of_interest != NULL);
    for (i = 0; i < self->number_of_variables; ++i)
      initial_solution[i] = 0.5
          * (self->smallest_values_of_interest[i] + self->largest_values_of_interest[i]);
  }
}

/* Commented to silence the compiler */
size_t coco_problem_get_suite_dep_index(coco_problem_t *problem) {
  return problem->suite_dep_index;
}


size_t coco_problem_get_suite_dep_function(coco_problem_t *problem) {
  return problem->suite_dep_function;
}

size_t coco_problem_get_suite_dep_instance(coco_problem_t *problem) {
  return problem->suite_dep_instance;
}

