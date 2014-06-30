#include "numbbo.h"

#include "numbbo_utilities.c"

/**
 * numbbo_allocate_problem(number_of_parameters):
 *
 * Allocate and pre-populate a new numbbo_problem_t for a problem with
 * ${number_of_parameters}.
 */
static numbbo_problem_t *numbbo_allocate_problem(const size_t number_of_parameters,
                                                 const size_t number_of_objectives,
                                                 const size_t number_of_constraints) {
    numbbo_problem_t *problem;
    problem = (numbbo_problem_t *)numbbo_allocate_memory(sizeof(numbbo_problem_t));
    /* Initialize fields to sane/safe defaults */
    problem->evaluate_function = NULL;
    problem->evaluate_constraint = NULL;
    problem->recommend_solutions = NULL;
    problem->free_problem = NULL;
    problem->number_of_parameters = number_of_parameters;
    problem->number_of_objectives = number_of_objectives;
    problem->number_of_constraints = number_of_constraints;
    problem->lower_bounds = numbbo_allocate_vector(number_of_parameters);
    problem->upper_bounds = numbbo_allocate_vector(number_of_parameters);
    problem->best_parameter = numbbo_allocate_vector(number_of_parameters);
    problem->best_value = numbbo_allocate_vector(number_of_objectives);
    problem->problem_name = NULL;
    problem->problem_id = NULL;
    return problem;
}

/**
 * Generic wrapper for a transformed numbbo_problem_t. 
 */
typedef struct numbbo_transformed_problem {
    numbbo_problem_t problem;  /* must be defined first to make inheritance / overload happen */
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
        free(obj->state);
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
        numbbo_allocate_memory(sizeof(numbbo_transformed_problem_t));
    numbbo_problem_t *problem = (numbbo_problem_t *)obj;

    problem->evaluate_function = _tfp_evaluate_function;
    problem->evaluate_constraint = _tfp_evaluate_constraint;
    problem->recommend_solutions = _tfp_recommend_solutions;
    problem->free_problem = _tfp_free_problem;
    problem->number_of_parameters = inner_problem->number_of_parameters;
    problem->number_of_objectives = inner_problem->number_of_objectives;
    problem->number_of_constraints = inner_problem->number_of_constraints;
    problem->lower_bounds = 
        numbbo_duplicate_vector(inner_problem->lower_bounds,
                                inner_problem->number_of_parameters);
    problem->upper_bounds = 
        numbbo_duplicate_vector(inner_problem->upper_bounds,
                                inner_problem->number_of_parameters);
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
