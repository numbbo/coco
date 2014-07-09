#include "numbbo.h"

#include "numbbo_utilities.c"

/**
 * _default_initial_solution(problem, initial_solution)
 *
 * Default implementation for the initial_solution method. The center
 * of ${problem}s region of interest is stored in the
 * vector pointed to by ${initial_solution}.
 */
static void _default_initial_solution(const numbbo_problem_t *problem, 
                                      double *initial_solution) {
    assert(problem != NULL);
    assert(problem->smallest_values_of_interest != NULL);
    assert(problem->largest_values_of_interest != NULL);
    for (size_t i = 0; i < problem->number_of_variables; ++i) 
        initial_solution[i] = 0.5 * (problem->smallest_values_of_interest[i] + problem->largest_values_of_interest[i]);
}

/**
 * numbbo_allocate_problem(number_of_variables):
 *
 * Allocate and pre-populate a new numbbo_problem_t for a problem with
 * ${number_of_variables}.
 */
static numbbo_problem_t *numbbo_allocate_problem(const size_t number_of_variables,
                                                 const size_t number_of_objectives,
                                                 const size_t number_of_constraints) {
    numbbo_problem_t *problem;
    problem = (numbbo_problem_t *)numbbo_allocate_memory(sizeof(numbbo_problem_t));
    /* Initialize fields to sane/safe defaults */
    problem->initial_solution = _default_initial_solution;
    problem->evaluate_function = NULL;
    problem->evaluate_constraint = NULL;
    problem->recommend_solutions = NULL;
    problem->free_problem = NULL;
    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = number_of_objectives;
    problem->number_of_constraints = number_of_constraints;
    problem->smallest_values_of_interest = numbbo_allocate_vector(number_of_variables);
    problem->largest_values_of_interest = numbbo_allocate_vector(number_of_variables);
    problem->best_parameter = numbbo_allocate_vector(number_of_variables);
    problem->best_value = numbbo_allocate_vector(number_of_objectives);
    problem->problem_name = NULL;
    problem->problem_id = NULL;
    return problem;
}

/**
 * Generic wrapper for a transformed numbbo_problem_t. 
 */
typedef struct numbbo_transformed_problem {
    /* must be defined first to make inheritance / overload happen */
    numbbo_problem_t problem;
    numbbo_problem_t *inner_problem;
    void *state;
} numbbo_transformed_problem_t;

static void _tfp_evaluate_function(numbbo_problem_t *self, double *x, double *y) {
    numbbo_transformed_problem_t *problem = (numbbo_transformed_problem_t *)self;
    assert(problem->inner_problem != NULL);
    numbbo_evaluate_function(problem->inner_problem, x, y);
}

static void _tfp_evaluate_constraint(numbbo_problem_t *self, double *x, double *y) {
    numbbo_transformed_problem_t *problem = (numbbo_transformed_problem_t *)self;
    assert(problem->inner_problem != NULL);
    numbbo_evaluate_constraint(problem->inner_problem, x, y);
}

static void _tfp_recommend_solutions(numbbo_problem_t *self,
                                     double *x, size_t number_of_solutions) {
    numbbo_transformed_problem_t *problem = (numbbo_transformed_problem_t *)self;
    assert(problem->inner_problem != NULL);
    numbbo_recommend_solutions(problem->inner_problem, x, number_of_solutions);
}

static void _tfp_free_problem(numbbo_problem_t *self) {
    numbbo_transformed_problem_t *obj = (numbbo_transformed_problem_t *)self;
    numbbo_problem_t *problem = (numbbo_problem_t *)obj;
    if (obj->inner_problem != NULL)
        numbbo_free_problem(obj->inner_problem);
    obj->inner_problem = NULL;
    if (obj->state != NULL)
        numbbo_free_memory(obj->state);
    /* Let the generic free problem code deal with the rest of the
     * fields. For this we clear the free_problem function pointer and
     * recall the generic function.
     */
    problem->free_problem = NULL;
    numbbo_free_problem(problem);
}

/**
 * numbbo_allocate_transformed_problem(inner_problem):
 *
 * Allocate a transformed problem that wraps ${inner_problem}. By
 * default all methods will dispatch to the ${inner_problem} method.
 *
 */
static numbbo_transformed_problem_t *
numbbo_allocate_transformed_problem(numbbo_problem_t *inner_problem) {
    numbbo_transformed_problem_t *obj =
        (numbbo_transformed_problem_t *)numbbo_allocate_memory(sizeof(numbbo_transformed_problem_t));
    numbbo_problem_t *problem = (numbbo_problem_t *)obj;

    problem->evaluate_function = _tfp_evaluate_function;
    problem->evaluate_constraint = _tfp_evaluate_constraint;
    problem->recommend_solutions = _tfp_recommend_solutions;
    problem->free_problem = _tfp_free_problem;
    problem->number_of_variables = inner_problem->number_of_variables;
    problem->number_of_objectives = inner_problem->number_of_objectives;
    problem->number_of_constraints = inner_problem->number_of_constraints;
    problem->smallest_values_of_interest = 
        numbbo_duplicate_vector(inner_problem->smallest_values_of_interest,
                                inner_problem->number_of_variables);
    problem->largest_values_of_interest = 
        numbbo_duplicate_vector(inner_problem->largest_values_of_interest,
                                inner_problem->number_of_variables);
    problem->best_value = 
        numbbo_duplicate_vector(inner_problem->best_value,
                                inner_problem->number_of_objectives);
    problem->best_parameter = 
        numbbo_duplicate_vector(inner_problem->best_parameter,
                                inner_problem->number_of_objectives);
    problem->problem_name = numbbo_strdup(inner_problem->problem_id);
    problem->problem_id = numbbo_strdup(inner_problem->problem_id);
    obj->inner_problem = inner_problem;
    obj->state = NULL;
    return obj;
}
