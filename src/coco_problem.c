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
    problem->alg_name = NULL;
    problem->data = NULL;
    return problem;
}

static coco_problem_t *coco_duplicate_problem(coco_problem_t *other) {
    size_t i;
    coco_problem_t *problem;
    problem = coco_allocate_problem(other->number_of_variables,
                                    other->number_of_objectives,
                                    other->number_of_constraints);

    problem->evaluate_function = other->evaluate_function;
    problem->evaluate_constraint = other->evaluate_constraint;
    problem->recommend_solutions = other->recommend_solutions;
    problem->free_problem = NULL;

    for (i = 0; i < problem->number_of_variables; ++i) {
        problem->smallest_values_of_interest[i] = other->smallest_values_of_interest[i];
        problem->largest_values_of_interest[i] = other->largest_values_of_interest[i];
        problem->best_parameter[i] = other->best_parameter[i];
    }

    for (i = 0; i < problem->number_of_objectives; ++i) {
        problem->best_value[i] = other->best_value[i];
    }

    problem->problem_name = coco_strdup(other->problem_name);
    problem->problem_id = coco_strdup(other->problem_id);
    return problem;
}

typedef void (*coco_transform_free_data_t) (void *data);

/**
 * Generic data member of a transformed (or "outer") coco_problem_t.
 */
typedef struct {
    coco_problem_t *inner_problem;
    void *data;
    coco_transform_free_data_t free_data;
} coco_transform_data_t;

static void _tfp_evaluate_function(coco_problem_t *self, double *x, double *y) {
    coco_transform_data_t *data;
    assert(self != NULL);
    assert(self->data != NULL);
    data = self->data;
    assert(data->inner_problem != NULL);

    coco_evaluate_function(data->inner_problem, x, y);
}

static void _tfp_evaluate_constraint(coco_problem_t *self, double *x, double *y) {
    coco_transform_data_t *data;
    assert(self != NULL);
    assert(self->data != NULL);
    data = self->data;
    assert(data->inner_problem != NULL);

    coco_evaluate_constraint(data->inner_problem, x, y);
}

static void _tfp_recommend_solutions(coco_problem_t *self,
                                     double *x, size_t number_of_solutions) {
    coco_transform_data_t *data;
    assert(self != NULL);
    assert(self->data != NULL);
    data = self->data;
    assert(data->inner_problem != NULL);

    coco_recommend_solutions(data->inner_problem, x, number_of_solutions);
}

static void _tfp_free_problem(coco_problem_t *self) {
    coco_transform_data_t *data;
    assert(self != NULL);
    assert(self->data != NULL);
    data = self->data;
    assert(data->inner_problem != NULL);

    if (data->inner_problem != NULL) {
        coco_free_problem(data->inner_problem);
        data->inner_problem = NULL;
    }
    if (data->data != NULL) {
        if (data->free_data != NULL) {
            data->free_data(data->data);
            data->free_data = NULL;
        }
        coco_free_memory(data->data);
        data->data = NULL;
    }
    /* Let the generic free problem code deal with the rest of the
     * fields. For this we clear the free_problem function pointer and
     * recall the generic function.
     */
    self->free_problem = NULL;
    coco_free_problem(self);
}

/**
 * coco_allocate_transformed_problem(inner_problem):
 *
 * Allocate a transformed problem that wraps ${inner_problem}. By
 * default all methods will dispatch to the ${inner_problem} method.
 *
 */
static coco_problem_t *
coco_allocate_transformed_problem(coco_problem_t *inner_problem,
                                  void *userdata,
                                  coco_transform_free_data_t free_data) {
    coco_transform_data_t *data;
    coco_problem_t *self;

    data = coco_allocate_memory(sizeof(*data));
    data->inner_problem = inner_problem;
    data->data = userdata;
    data->free_data = free_data;

    self = coco_duplicate_problem(inner_problem);
    self->evaluate_function = _tfp_evaluate_function;
    self->evaluate_constraint = _tfp_evaluate_constraint;
    self->recommend_solutions = _tfp_recommend_solutions;
    self->free_problem = _tfp_free_problem;
    self->data = data;
    return self;
}

static void *coco_get_transform_data(coco_problem_t *self) {
    assert(self != NULL);
    assert(self->data != NULL);
    assert(((coco_transform_data_t *)self->data)->data != NULL);

    return ((coco_transform_data_t *)self->data)->data;
}

static coco_problem_t *coco_get_transform_inner_problem(coco_problem_t *self) {
    assert(self != NULL);
    assert(self->data != NULL);
    assert(((coco_transform_data_t *)self->data)->inner_problem != NULL);

    return ((coco_transform_data_t *)self->data)->inner_problem;
}
