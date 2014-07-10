/*
 * Internal NumBBO structures and typedefs.
 *
 * These are used throughout the NumBBO code base but should not be
 * used by any external code.
 */

#ifndef __NUMBBO_INTERNAL__ 
#define __NUMBBO_INTERNAL__

typedef void (*numbbo_initial_solution_function_t) (const struct numbbo_problem *self,
                                                    double *y);
typedef void (*numbbo_evaluate_function_t) (struct numbbo_problem *self,
                                            double *x, double *y);
typedef void (*numbbo_recommendation_function_t) (struct numbbo_problem *self,
                                                  double *x,
                                                  size_t number_of_solutions);

typedef void (*numbbo_free_function_t) (struct numbbo_problem *self);

/**
 * Description of a NUMBBO problem (instance)
 *
 * evaluate and free are opaque pointers which should not be called
 * directly. Instead they are used by the numbbo_* functions in
 * numbbo_generics.c. This indirection gives us the flexibility to add
 * generic checks or more sophisticated dispatch methods later on.
 *
 * Fields:
 *
 * number_of_variables - Number of parameters expected by the
 *   function and contraints.
 *
 * number_of_objectives - Number of objectives.
 *
 * number_of_constraints - Number of constraints.
 *
 * smallest_values_of_interest, largest_values_of_interest - Vectors
 *   of length 'number_of_variables'. Lower/Upper bounds of the 
 *   region of interest.
 *
 * problem_name - Descriptive name for the test problem. May be NULL
 *   to indicate that no name is known.
 *
 * problem_id - Short name consisting of _only_ [a-z0-9], '-' and '_'
 *   that, when not NULL, must be unique. It can for example be used
 *   to generate valid directory names under which to store results.
 */
struct numbbo_problem {
    numbbo_initial_solution_function_t initial_solution;
    numbbo_evaluate_function_t evaluate_function;
    numbbo_evaluate_function_t evaluate_constraint;
    numbbo_recommendation_function_t recommend_solutions;
    numbbo_free_function_t free_problem;
    size_t number_of_variables;
    size_t number_of_objectives;
    size_t number_of_constraints;
    double *smallest_values_of_interest;
    double *largest_values_of_interest;
    double *best_value;
    double *best_parameter;
    char *problem_name;
    char *problem_id;
};

#endif
