#include <assert.h>
#include <stddef.h>

#include "coco.h"
#include "coco_internal.h"

/**
 * Evaluates the problem function, increases the number of evaluations and updates the best observed value
 * and the best observed evaluation number.
 *
 * @note Both x and y must point to correctly sized allocated memory regions.
 *
 * @param problem The given COCO problem.
 * @param x The decision vector.
 * @param y The objective vector that is the result of the evaluation (in single-objective problems only the
 * first vector item is being set).
 */
void coco_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  /* implements a safer version of self->evaluate(self, x, y) */
  assert(problem != NULL);
  assert(problem->evaluate_function != NULL);
  problem->evaluate_function(problem, x, y);
  problem->evaluations++; /* each derived class has its own counter, only the most outer will be visible */
#if 1
  /* A little bit of bookkeeping */
  if (y[0] < problem->best_observed_fvalue[0]) {
    problem->best_observed_fvalue[0] = y[0];
    problem->best_observed_evaluation[0] = problem->evaluations;
  }
#endif
}

size_t coco_problem_get_evaluations(coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->evaluations;
}

#if 1  /* tentative */
double coco_problem_get_best_observed_fvalue1(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->best_observed_fvalue[0];
}

/**
 * @note This function breaks the black-box property: the returned  value is not meant to be used by the
 * optimization algorithm other than for testing termination conditions.
 */
double coco_problem_get_final_target_fvalue1(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->best_value != NULL);
  assert(problem->final_target_delta != NULL);
  return problem->best_value[0] + problem->final_target_delta[0];
}
#endif

/**
 * @note None of the problems implement this function yet!
 * @note Both x and y must point to correctly sized allocated memory regions.
 *
 * @param problem The given COCO problem.
 * @param x The decision vector.
 * @param y The vector of constraints that is the result of the evaluation.
 */
void coco_evaluate_constraint(coco_problem_t *problem, const double *x, double *y) {
  /* implements a safer version of problem->evaluate(problem, x, y) */
  assert(problem != NULL);
  if (problem->evaluate_constraint == NULL) {
    coco_error("coco_evaluate_constraint(): No constraint function implemented for problem %s",
        problem->problem_id);
  }
  problem->evaluate_constraint(problem, x, y);
}

/**
 * @note None of the observers implements this function yet!
 * @note Both x and y must point to correctly sized allocated memory regions.
 * @note number_of_solutions is expected to be larger than 1 only for multi-objective problems.

 * @param problem The given COCO problem.
 * @param x The array of vector.
 * @param y The vector of constraints that is the result of the evaluation.
 */
void coco_recommend_solutions(coco_problem_t *problem, const double *x, size_t number_of_solutions) {
  assert(problem != NULL);
  if (problem->recommend_solutions == NULL) {
    coco_error("coco_recommend_solutions(): No recommend solution function implemented for problem %s",
        problem->problem_id);
  }
  problem->recommend_solutions(problem, x, number_of_solutions);
}

/**
 * @note Do not modify the returned string! If you free the problem, the returned pointer becomes invalid.
 * When in doubt, use coco_strdup() on the returned value.
 */
const char *coco_problem_get_name(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->problem_name != NULL);
  return problem->problem_name;
}

/**
 * The ID is guaranteed to contain only characters in the set [a-z0-9_-]. It should therefore be safe to use
 * it to construct filenames or other identifiers.
 *
 * Each problem ID should be unique within each benchmark suite.
 *
 * @note Do not modify the returned string! If you free the problem, the returned pointer becomes invalid.
 * When in doubt, use coco_strdup() on the returned value.
 */
const char *coco_problem_get_id(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->problem_id != NULL);
  return problem->problem_id;
}

const char *coco_problem_get_type(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->problem_type != NULL);
  return problem->problem_type;
}

size_t coco_problem_get_dimension(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->number_of_variables > 0);
  return problem->number_of_variables;
}

size_t coco_problem_get_number_of_objectives(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->number_of_objectives > 0);
  return problem->number_of_objectives;
}

size_t coco_problem_get_number_of_constraints(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->number_of_constraints;
}

const double *coco_problem_get_smallest_values_of_interest(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->smallest_values_of_interest != NULL);
  return problem->smallest_values_of_interest;
}

const double *coco_problem_get_largest_values_of_interest(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->largest_values_of_interest != NULL);
  return problem->largest_values_of_interest;
}

/**
 * If a special method for setting an initial solution to the problem does not exist, the center of the
 * problem's region of interest is the initial solution.
 * @param problem The given COCO problem.
 * @param initial_solution The pointer to the initial solution being set by this method.
 */
void coco_problem_get_initial_solution(const coco_problem_t *problem, double *initial_solution) {
  assert(problem != NULL);
  if (problem->initial_solution != NULL) {
    problem->initial_solution(problem, initial_solution);
  } else {
    size_t i;
    assert(problem->smallest_values_of_interest != NULL);
    assert(problem->largest_values_of_interest != NULL);
    for (i = 0; i < problem->number_of_variables; ++i)
      initial_solution[i] = 0.5
          * (problem->smallest_values_of_interest[i] + problem->largest_values_of_interest[i]);
  }
}

size_t coco_problem_get_suite_dep_index(coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->suite_dep_index;
}

size_t coco_problem_get_suite_dep_function(coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->suite_dep_function > 0);
  return problem->suite_dep_function;
}

size_t coco_problem_get_suite_dep_instance(coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->suite_dep_instance > 0);
  return problem->suite_dep_instance;
}

