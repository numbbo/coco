#include "coco.h"

#include "coco_utilities.c"

/**
 * coco_allocate_problem(number_of_variables):
 *
 * Allocate and pre-populate a new coco_problem_t for a problem with
 * ${number_of_variables}.
 */
static coco_problem_t *coco_allocate_problem(const size_t number_of_variables,
                                                 const size_t number_of_objectives,
                                                 const size_t number_of_constraints) {
    coco_problem_t *problem;
    problem = (coco_problem_t *)coco_allocate_memory(sizeof(coco_problem_t));
    /* Initialize fields to sane/safe defaults */
    problem->initial_solution = NULL;
    problem->evaluate_function = NULL;
    problem->evaluate_constraint = NULL;
    problem->recommend_solutions = NULL;
    problem->free_problem = NULL;
    problem->number_of_variables = number_of_variables;
    problem->number_of_objectives = number_of_objectives;
    problem->number_of_constraints = number_of_constraints;
    problem->smallest_values_of_interest = coco_allocate_vector(number_of_variables);
    problem->largest_values_of_interest = coco_allocate_vector(number_of_variables);
    problem->best_parameter = coco_allocate_vector(number_of_variables);
    problem->best_value = coco_allocate_vector(number_of_objectives);
    problem->problem_name = NULL;
    problem->problem_id = NULL;
    return problem;
}

/**
 * Generic wrapper for a transformed (or "outer") coco_problem_t.
 */
typedef struct coco_transformed_problem {
    /* problem_t must be defined first to make inheritance / overload happen */
    coco_problem_t problem;
    coco_problem_t *inner_problem;
    void *state;
} coco_transformed_problem_t;

static void _tfp_evaluate_function(coco_problem_t *self, double *x, double *y) {
    coco_transformed_problem_t *problem = (coco_transformed_problem_t *)self;
    assert(problem->inner_problem != NULL);
    coco_evaluate_function(problem->inner_problem, x, y);
}

static void _tfp_evaluate_constraint(coco_problem_t *self, double *x, double *y) {
    coco_transformed_problem_t *problem = (coco_transformed_problem_t *)self;
    assert(problem->inner_problem != NULL);
    coco_evaluate_constraint(problem->inner_problem, x, y);
}

static void _tfp_recommend_solutions(coco_problem_t *self,
                                     double *x, size_t number_of_solutions) {
    coco_transformed_problem_t *problem = (coco_transformed_problem_t *)self;
    assert(problem->inner_problem != NULL);
    coco_recommend_solutions(problem->inner_problem, x, number_of_solutions);
}

static void _tfp_free_problem(coco_problem_t *self) {
    coco_transformed_problem_t *obj = (coco_transformed_problem_t *)self;
    coco_problem_t *problem = (coco_problem_t *)obj;
    if (obj->inner_problem != NULL)
        coco_free_problem(obj->inner_problem);
    obj->inner_problem = NULL;
    if (obj->state != NULL)
        coco_free_memory(obj->state);
    /* Let the generic free problem code deal with the rest of the
     * fields. For this we clear the free_problem function pointer and
     * recall the generic function.
     */
    problem->free_problem = NULL;
    coco_free_problem(problem);
}

/**
 * coco_allocate_transformed_problem(inner_problem):
 *
 * Allocate a transformed problem that wraps ${inner_problem}. By
 * default all methods will dispatch to the ${inner_problem} method.
 *
 */
static coco_transformed_problem_t *
coco_allocate_transformed_problem(coco_problem_t *inner_problem) {
    coco_transformed_problem_t *obj =
        (coco_transformed_problem_t *)coco_allocate_memory(sizeof(coco_transformed_problem_t));
    coco_problem_t *problem = (coco_problem_t *)obj;

    problem->evaluate_function = _tfp_evaluate_function;
    problem->evaluate_constraint = _tfp_evaluate_constraint;
    problem->recommend_solutions = _tfp_recommend_solutions;
    problem->free_problem = _tfp_free_problem;
    problem->number_of_variables = inner_problem->number_of_variables;
    problem->number_of_objectives = inner_problem->number_of_objectives;
    problem->number_of_constraints = inner_problem->number_of_constraints;
    problem->smallest_values_of_interest = 
        coco_duplicate_vector(inner_problem->smallest_values_of_interest,
                                inner_problem->number_of_variables);
    problem->largest_values_of_interest = 
        coco_duplicate_vector(inner_problem->largest_values_of_interest,
                                inner_problem->number_of_variables);
    problem->best_value = 
        coco_duplicate_vector(inner_problem->best_value,
                                inner_problem->number_of_objectives);
    problem->best_parameter = 
        coco_duplicate_vector(inner_problem->best_parameter,
                                inner_problem->number_of_objectives);
    problem->problem_name = coco_strdup(inner_problem->problem_id);
    problem->problem_id = coco_strdup(inner_problem->problem_id);
    obj->inner_problem = inner_problem;
    obj->state = NULL;
    return obj;
}
